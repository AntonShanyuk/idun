# Data layout

§1–§3 describe the packed cache on disk, §4–§6 what a fit keeps in memory. The cache format is versioned; any change
bumps `format_version` and invalidates old caches loudly.

## 1. Source panel (Numerai v6, read-only)

Numerai's v6 `train.parquet` and `validation.parquet` (`idun_numerai.download`):

- the rows with a value of the chosen target (`build_cache(..., target=)`), in file order, then a stable sort by date;
- about 6.9 million rows from 2003 on (the validation file grows every round).

Features (1,040):

| group | count | raw type | stored as |
|---|---|---|---|
| obfuscated | 1,017 | int8, values exactly 0..4 (asserted) | 4-bit codes, no binning |
| named factors | 22 | float32 with NaN | uint8 bins per refit: 0..254 real, 255 = missing |
| `feature_country` | 1 | string, 47 categories | uint8 code, index into the sorted category list |

Feature order: named (sorted) + obfuscated (sorted) + country.

## 2. Packed cache v1

Written by `idun.data.PackedWriter` (Numerai's data through `idun_numerai.data.build_cache`), read by
`idun.data.PackedDataset`. `n` = 6,947,317 rows.

| file | dtype / layout | size |
|---|---|---|
| `manifest.json` | format name and version; source directory and fingerprints (name, bytes, `mtime_ns` of `train.parquet`, `validation.parquet`, `features.json`); builder versions (idun_numerai, numpy, pyarrow); seed, sample size, purge; `n_rows`; column strides; feature lists; country list; refit table; per-binning bin counts; size and sha256 of every array file | small |
| `obf.u4` | 1,017 columns, column-major. Column `f` starts at `f × stride`, `stride = round_up(ceil(n/2), 16 KiB)`. Byte `i` holds row `2i` in the low nibble and row `2i+1` in the high nibble; padding is zero | 3.55 GB |
| `named.f32` | 22 float32 columns of raw values (NaN kept), `stride = round_up(4n, 16 KiB)` | 611 MB |
| `named.r{k}.u8` | 22 uint8 columns (`stride = round_up(n, 16 KiB)`): every row binned with refit `k`'s edges | 153 MB each |
| `named.r{k}.edges.f32` | 22 × 255 float32 upper edges, **+inf padded** past each feature's `n_bins − 1` real edges | 22 KB |
| `country.u8` | 1 byte per row: index into the sorted country list | 7 MB |
| `y.f32` | the target | 28 MB |
| `era.i32` | date as days since 1970-01-01 | 28 MB |
| `ticker.i32` + `tickers.json` | index into the sorted unique ticker list | 28 MB |

Design notes:

- **Column-major** so a tree reads only its sampled columns, each one contiguous.
- **Pairs of rows per byte, not pairs of features.** Packing two features per byte would drag the unsampled partner
  feature through the memory bus. Two rows per byte keeps each feature's column self-contained.
- **16 KiB-aligned columns.** One page per alignment unit: the whole file maps page-aligned, Metal wraps it without a
  copy (verified), and every column starts on a 16-byte boundary for vector loads.
- **Raw named values are kept** (`named.f32`) so binnings can be refitted without the parquet files, and so a LightGBM
  baseline (LightGBM bins the raw values itself) and a ridge regression can stream from the cache.
- **Binning per refit.** Edges come from a 200,000-row sample of the refit's own training window, drawn
  without replacement by `numpy.random.Generator(PCG64([seed, refit]))` (`bin_construct_sample_cnt` analogue), so later
  rows never leak into the bins. Per feature, the distinct finite sample values: with ≤ 255 of them each gets its own
  bin; otherwise the cuts sit at the `i/255` count quantiles (`i = 1..254`, duplicates merged). Each edge is the float32
  midpoint between the two distinct values it separates; if that midpoint rounds onto the upper value, the lower value
  is used, so adjacent values never merge. A value `v` maps to the first bin whose upper edge is ≥ `v` (the +inf padding
  ends every row), and NaN maps to 255. This is plain quantile binning, not LightGBM's `BinMapper`.
- **Staleness.** The reader recomputes the source fingerprints and refuses a stale cache, and checks every array's size.
- **Copies.** On a machine without the source files (a copied cache), `IDUN_CACHE_COPY=1` replaces the source check
  with the checksum of every array file (a few seconds for the v6 cache): a copy is never opened unchecked.
  `verify` recomputes the checksums. It never rebuilds silently.

### Build (streaming, two passes)

1. **Rows:** `date`, `numerai_ticker`, `feature_country` and the target of every batch. Rows without a target are
   dropped. The rows must already be in date order, as the v6 files are, so the stable sort by date is a no-op; the
   builder refuses unsorted input rather than re-sorting it.
2. **Features:** the same rows' 1,017 obfuscated values (checked to be exactly 0..4) and 22 named
   values, staged in 262,144-row column-major blocks and written with `pwrite` at each column's offset.
3. Per refit, edges from the sample of its window, then every row binned. Checksums, then the manifest. Everything is
   written to `<out>.partial/`, renamed to `<out>/` only when complete, and deleted if anything fails.

## 3. Walk-forward windows

The manifest lists the windows (`idun.data.Refit`), as row indices into the date-sorted panel: window `k` trains on rows
`[0, train_hi)` and predicts rows `[pred_lo, pred_hi)`, with a purge of eras between the two. Which dates the windows
start on, and the purge, are the builder's arguments (`idun_numerai.data.build_cache(..., refit_dates=, purge=)`;
`idun_numerai.refits.refit_table`).

## 4. Per-fit state (shared buffers)

| array | dtype | meaning |
|---|---|---|
| `pred` | float32[n] | running prediction, starting at 0 |
| `g` | float32[n] | `pred − y` for this iteration |
| `q` | int8[n] | quantized gradient (in-bag rows only; others 0) |
| `bag` | uint8[n] | 1 if the row is in this iteration's bag |
| `node` | uint8[n] | depth-wise: node index within the current level, 0..2^d − 1. Leaf-wise: the row's leaf id, 0..max_leaves − 1 |

## 5. Histograms

`hist[node_slot, bin_offset[f] + b] = (int32 sum_q, int32 count)`, flattened over the sampled features. Here
`bin_offset` is the prefix sum of each sampled feature's bin count: 5 for obfuscated features, up to 256 for named ones,
and 47 for country. Only the smaller child of each split occupies a slot; its sibling is derived by subtraction.

## 6. Trees: complete-heap layout

Depth-wise trees with `max_depth = D` (6 by default) are stored as complete binary heaps: `2^D − 1`
internal nodes and `2^D` leaves.

| field (per internal node) | dtype | meaning |
|---|---|---|
| `kind` | uint8 | 0 = pass-through (the node stopped splitting; all rows go left), 1 = numeric, 2 = categorical |
| `feature` | int32 | feature index in the data's feature order |
| `threshold` | uint8 | numeric: bins ≤ threshold go left |
| `missing_left` | uint8 | named features: where bin 255 goes |
| `cat_left` | uint64 | categorical: bitmask of category codes that go left (47 ≤ 64) |
| `leaf_value` (per leaf) | float32 | learning rate included. A node that stopped early has its value copied to the leaf reached by always going left |
| `leaf_count` (per leaf) | int32 | the in-bag training rows that reached the leaf (TreeSHAP's covers; optional); 0 for a leaf no row reaches |

Traversal: `node = 0`, then `D` times `node = 2·node + 1 + go_right`, then `leaf = node − (2^D − 1)`. The loop has a fixed
trip count and no early exit, which suits SIMD execution, and the layout also fits symmetric (oblivious) trees later.

### 6.1 Leaf-wise trees: node lists

Leaf-wise trees (`growth="leafwise"`, `idun.gbm.tree.NodeForest`) have no depth bound, so they are stored as
LightGBM stores trees: `max_leaves − 1` internal nodes with the fields above plus two int16 child pointers, `left` and
`right` (an internal node when ≥ 0, else leaf `~child`), and `max_leaves` leaf values indexed by leaf id. The root is
node 0. The tree's `m`-th split, of leaf `l`, becomes node `m`: its left child is leaf `l`, its right child leaf
`m + 1`, and the pointer that led to leaf `l` now leads to node `m`. Every child node therefore comes after its
parent, so a walk (`node = 0`; follow `left` or `right` until a child is negative) ends within `max_leaves − 1` steps;
`Booster.load` checks it. A tree that stopped early leaves its last nodes unused; a tree that never split keeps a
pass-through root whose children are both leaf 0.

During training `node[i]` holds row `i`'s leaf id (uint8, so at most 256 leaves), which is also the index of its leaf
sums and leaf value.
