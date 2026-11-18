# Model file format (`.igbm`, v3)

Written by `idun.gbm.Booster.save`, read by `Booster.load`. LightGBM model-format compatibility is out of scope.

A `.igbm` file is an uncompressed NumPy `.npz` archive. It must load with `np.load(path, allow_pickle=False)`: no object
arrays, no pickles. `Booster.load` refuses any file that is not exactly this layout.

| entry | dtype, shape | meaning |
|---|---|---|
| `meta` | uint8[*] | UTF-8 JSON, see below |
| `kind` | uint8[T, I] | 0 pass-through, 1 numeric, 2 categorical |
| `feature` | int32[T, I] | feature index (the data's feature order) |
| `threshold` | uint8[T, I] | numeric split: bins ≤ threshold go left |
| `missing_left` | uint8[T, I] | direction of the missing bin (255) |
| `cat_left` | uint64[T, I] | categorical split: bitmask of codes that go left |
| `left`, `right` | int16[T, I] | leaf-wise models only: child pointers, an internal node when ≥ 0, else leaf `~child` |
| `leaf_value` | float32[T, L] | learning rate included |
| `leaf_count` | int32[T, L] | optional: each leaf's in-bag training rows (TreeSHAP's covers) |
| `named_edges` | float32[N, 255] | the binning's upper edges for the N numeric features (+inf padded) |
| `named_n_bins` | uint16[N] | real bins per numeric feature |

Depth-wise models (`params.growth == "depthwise"`) store complete heaps: `I = 2^D − 1` internal nodes and `L = 2^D`
leaves for `D = max_depth`. Leaf-wise models store node lists: `I = max_leaves − 1`, `L = max_leaves`. Tree layout and
traversal: [data-layout.md](../architecture/data-layout.md) §6.

`meta` JSON:

```json
{
  "format": "idun-gbm",
  "format_version": 3,
  "idun_version": "0.0.0",
  "params": {"...": "Params as a dict"},
  "feature_names": ["..."],
  "categories": [["..."]],
  "binning": "r0",
  "growth": "depthwise",
  "max_depth": 6,
  "n_trees": 100,
  "train_rows": [0, 100000],
  "backend": "metal",
  "created": "...UTC time of the save, ISO 8601",
  "best_iteration": null,
  "evals": {"valid_0": ["...one score after each tree trained"]}
}
```

Rules:

- `format_version` changes on any incompatible change. Loading an unknown version raises. Version 3 adds
  `params.growth` and the leaf-wise node lists; a version-1 or 2 file has no `growth` and loads as depth-wise.
  Version 2 stores every categorical column's levels as `categories`; a version-1 file's `countries` loads as the one
  column.
- A leaf-wise model loads only if every walk ends at a leaf: each child node comes after its parent and every leaf id
  is below `max_leaves` (`NodeForest.check`).
- `predict` checks `binning`, `feature_names` and `categories` against the dataset and raises on mismatch.
- The model records idun's version (`idun_version`), not a git commit.
- `best_iteration` (early stopping's best tree, or null) and `evals` (each validation set's scores) are optional keys:
  a file without them loads with null and no scores.
- `leaf_count` is an optional entry. A file without it loads and predicts, but has no contributions
  (`predict_contributions` raises).
- Leaf values are stored exactly as trained, so a loaded model predicts bit-identically to the in-memory one (unit test).
