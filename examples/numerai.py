"""Numerai with idun-numerai: download the data, pack it, train idun_gbm on a walk-forward window and score it per era.

    uv run --frozen python examples/numerai.py WORKDIR

1. `download` fetches Numerai's v6 training and validation files and the feature list (public data, no credentials)
   into WORKDIR/v6: about 3.6 GB, more as the validation file grows each round.
2. `build_cache` packs them into idun's memory-mapped format in WORKDIR/cache (about 4.5 GB), streaming the parquet
   files so that the panel never sits in memory whole: the rows of one target (``TARGET``, Numerai's main target
   here) and one walk-forward window, which trains on every era up to ``PURGE`` eras before ``REFIT`` and predicts
   from it. Each step is skipped when its folder exists. Choose the target, the dates and the purge for your own
   evaluation; the purge must cover the target's horizon.
3. `gbm.train` fits idun_gbm on the window's training rows on Metal, with a few of its parameters set
   (docs/learners/gbm.md lists them all); they are an illustration, not a tuned configuration.
4. `per_era(corr, ...)` scores the predictions of the window's predicted rows, one correlation per era.
   `idun_numerai.score.ncorr` needs each era's rows of Numerai's `validation_neutralizer.parquet` as well.
5. `upload.predictor` and `upload.save` write the model as Numerai's model upload, WORKDIR/idun_gbm.pkl: a predict
   function that runs on Numerai's servers with NumPy alone. The pickle must come from the runtime's Python (3.12).
"""

import argparse
import sys
from pathlib import Path

from idun import gbm
from idun.data import PackedDataset
from idun_numerai import upload
from idun_numerai.data import build_cache
from idun_numerai.download import download
from idun_numerai.refits import to_date
from idun_numerai.score import corr, per_era

TARGET = "target"  # the column the cache keeps (Numerai's main target)
REFIT = ("2023-01-06",)  # one window: train before it, predict from it on
PURGE = 12  # eras left out between the training rows and the window's first predicted era
PARAMS = gbm.Params(num_trees=300, feature_fraction=0.5, seed=0)


def main() -> None:
    """Download, pack, train, predict and score."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path)
    workdir = parser.parse_args().workdir
    source, cache = workdir / "v6", workdir / "cache"

    if not source.exists():
        download(source)
    if not cache.exists():
        build_cache(source, cache, seed=0, target=TARGET, refit_dates=REFIT, purge=PURGE)

    # Each walk-forward window bins the named features on its own training rows: the first is "r0".
    data = PackedDataset.open(cache, binning="r0")
    window = data.refits[0]
    print(f"{data.n_rows:,} rows, {len(data.feature_names):,} features; train on rows [0, {window.train_hi:,}), "
          f"predict [{window.pred_lo:,}, {window.pred_hi:,}) from {window.date}")  # fmt: skip

    model = gbm.train(PARAMS, data, rows=slice(0, window.train_hi), backend="metal")
    rows = slice(window.pred_lo, window.pred_hi)
    pred = model.predict(data, rows=rows, backend="metal")
    model.save(workdir / "idun_gbm.igbm")

    scores = per_era(corr, data.era[rows], pred, data.target[rows])
    first, last = to_date(int(data.era[window.pred_lo])), to_date(int(data.era[window.pred_hi - 1]))
    print(f"per-era correlation over {scores.size} eras from {first} to {last}: mean {scores.mean():.4f}")

    predict = upload.predictor([model], data)
    if sys.version_info[:2] == upload.RUNTIME_PYTHON:
        print(f"model upload: {upload.save(predict, workdir / 'idun_gbm.pkl')}")
    else:
        print(
            f"model upload: not written; rerun under Python {'.'.join(map(str, upload.RUNTIME_PYTHON))}, the runtime's"
        )


if __name__ == "__main__":
    main()
