"""idun's three learners on a synthetic table: fit, predict, compare backends, save and load.

    uv run --frozen python examples/quickstart.py

Needs an Apple silicon Mac (Metal); on other machines pass ``backend="cpu"`` to idun_gbm and leave out the networks,
whose CPU backend is a float64 reference meant for tests. Runs in about a minute on an M1.
"""

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import idun
from idun.gbm import GBMRegressor
from idun.realmlp import RealMLPRegressor
from idun.tabm import TabMClassifier


def table(n_rows: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return eight numeric columns, one in ten values missing, and a noisy non-linear target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 8))
    y = np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2] + 0.5 * np.abs(x[:, 3]) + rng.normal(scale=0.3, size=n_rows)
    x[rng.random(x.shape) < 0.1] = np.nan
    return x, y


def main() -> None:
    """Fit idun_gbm, idun_tabm and idun_realmlp and print their test scores."""
    x_train, y_train = table(20_000, seed=0)
    x_test, y_test = table(5_000, seed=1)

    # idun_gbm: LightGBM's model. The CPU and Metal backends build the same trees, bit for bit.
    gbm = GBMRegressor(num_trees=300, learning_rate=0.05).fit(x_train, y_train)
    cpu = GBMRegressor(num_trees=300, learning_rate=0.05, backend="cpu").fit(x_train, y_train)
    same = np.array_equal(gbm.predict(x_test), cpu.predict(x_test))
    print(f"idun_gbm     R² {gbm.score(x_test, y_test):.4f}  (CPU and Metal bit-identical: {same})")

    # Fit again with the same seed: the same bits, run after run.
    again = GBMRegressor(num_trees=300, learning_rate=0.05).fit(x_train, y_train)
    print(f"idun_gbm     refit bit-identical: {np.array_equal(gbm.predict(x_test), again.predict(x_test))}")

    # idun_tabm as a classifier: any labels, two classes or more.
    labels = np.where(y_train > np.median(y_train), "high", "low")
    tabm = TabMClassifier(epochs=10).fit(x_train, labels)
    truth = np.where(y_test > np.median(y_train), "high", "low")
    print(f"idun_tabm    accuracy {tabm.score(x_test, truth):.4f}, classes {tabm.classes_.tolist()}")

    # idun_realmlp as a regressor, a bag of two seeds (it predicts their mean).
    mlp = RealMLPRegressor(epochs=20, n_seeds=2).fit(x_train, y_train)
    print(f"idun_realmlp R² {mlp.score(x_test, y_test):.4f}")

    # Every estimator saves to one file; idun.load reads any of them back.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gbm.idun"
        gbm.save(path)
        loaded = idun.load(path)
        same = np.array_equal(loaded.predict(x_test), gbm.predict(x_test))
        print(f"loaded {type(loaded).__name__} from {path.name}: same predictions {same}")


if __name__ == "__main__":
    main()
