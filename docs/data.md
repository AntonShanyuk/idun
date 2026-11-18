# Data

idun reads three kinds of input: a table in memory (`idun.Dataset` for idun_gbm, a matrix for the networks), and a
packed dataset on disk for tables that do not fit in memory. The estimators take a NumPy array, a pandas DataFrame or
an Arrow table and build what they need themselves; the lower-level functions take these objects directly.

## `idun.Dataset`: a table binned for idun_gbm

```python
import idun

data = idun.Dataset(X, y, categorical=["sector"], codes=["rating"])  # names, or positions for an array
test = data.transform(X_test)  # the same edges and levels; an unseen level raises
part = data.subset(slice(0, 1000))  # rows, as a slice or indices
```

Every column is numeric unless you name it as codes or categorical; nothing is chosen from the data.

| kind | stored as | rules |
|---|---|---|
| numeric | uint8 quantile bins, at most `max_bin`, and bin 255 for a missing value | edges from a seeded sample of `subsample_for_bin` rows, or from `bin_rows` |
| codes | 4-bit codes, two rows per byte | integers 0 to 4, no missing value: half the memory of a numeric column and faster histograms |
| categorical | one uint8 code per row over the sorted levels | at most 64 levels, no missing value |

A model remembers the binning it was trained on (`Dataset.binning`, an id of the edges and levels) and refuses data
binned another way, so predict on `data.transform(X_test)` or pass raw rows to an estimator.

| parameter | default | what it does |
|---|---|---|
| `categorical` | `()` | the categorical columns, by name or position |
| `codes` | `()` | the 4-bit code columns, by name or position |
| `feature_names` | `None` | names for an array's columns (a DataFrame or a table has its own) |
| `bin_rows` | `None` | the rows the numeric bin edges come from, instead of a sample |
| `seed` | `0` | the bin sample's seed |
| `max_bin` | `255` | the most bins of a numeric column, 2 to 255; fewer bins coarsen the columns but do not make a fit faster |
| `subsample_for_bin` | `200000` | the size of the bin sample (LightGBM's name); not with `bin_rows` |

## Matrices for the networks

The networks read float32 inputs. Their estimators encode a table themselves (`idun.nn.table.TableEncoder`: numeric
columns standardized for idun_tabm and left to RealMLP's own scaling, categorical columns one-hot encoded or, for
RealMLP, embedded), and the lower-level `train_tabm` and `train_realmlp` take an `idun.nn.inputs.MatrixData(x, y,
n_classes)`: a finite float32 `(n_rows, n_inputs)` matrix and one label per row, a class or, with `n_classes=1`, a
regression target.

## Packed datasets: tables larger than memory

A packed dataset is a date-sorted panel stored column by column on disk and memory-mapped, so a fit reads it without
loading it, and Metal wraps its arrays without a copy. It holds numeric columns (raw values, and their bins per
walk-forward window), 4-bit code columns two rows per byte, one categorical column, a target, a time key and an entity
id per row, and its walk-forward windows (`idun.data.Refit`: train on rows `[0, train_hi)`, predict rows
`[pred_lo, pred_hi)`).

- `idun.data.PackedWriter` writes one from blocks of columns, streaming them to disk at each column's offset, so a
  table far larger than memory passes through; it bins each window's numeric columns on a sample of that window's
  training rows and checksums every array.
- `idun.data.PackedDataset.open(path, binning=...)` maps it read-only with one window's binning, and refuses a dataset
  that is stale, truncated or of another format, never rebuilding silently. Its fields keep the names of the panel it
  was first built for: `named_*` are the numeric columns, `obfuscated` the code columns, `country` the categorical
  column, `era` the time key in days and `ticker` the entity id.
- idun_gbm, idun_tabm and idun_realmlp train on it directly (`rows=slice(0, window.train_hi)`), the networks reading
  their labels from a page-aligned class or target array (`idun.nn.inputs.class_array`, `target_array`).
- `idun.data.lag_index(time, group, lags, step)` finds each row's earlier rows of the same entity, for features over
  time.

The format is on the [data layout page](architecture/data-layout.md). `idun-numerai` builds one from Numerai's files
(`idun_numerai.data.build_cache`).
