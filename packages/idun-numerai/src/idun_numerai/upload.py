"""Numerai's model uploads: trained models as a predict function that needs NumPy alone.

Numerai runs an uploaded model as a cloudpickled ``predict(live_features, live_benchmark_models)`` returning a
``prediction`` column, on Linux with one CPU in a fixed environment (numerai-predict's Python 3.12 image: NumPy 2.1,
pandas, cloudpickle; not idun). `predictor` turns models trained on a packed Numerai dataset into such a function. It
carries each model as arrays and predicts with NumPy: no idun, numba or Metal library is needed where it runs.

- idun_gbm (`idun.gbm.Booster`): the live named features are binned with the model's edges, the codes and the
  country read as the packed dataset stores them, and the trees walked with float32 sums in tree order, so the
  predictions are `Booster.predict`'s, bit for bit.
- idun_tabm (`idun.tabm.TabMModel`) and idun_realmlp (`idun.realmlp.RealMLPModel`), single-target models: the inputs
  as `idun.nn.inputs.dense_inputs` builds them and the forward pass of the float64 ``"reference"`` backend, so the
  predictions are that backend's up to float64 rounding (and Metal's within float32 rounding): the expected class, the
  output (``"mse"``) or, for an HL-Gauss TabM, the expected bin centre.
- A bag (a list of models of one kind) predicts the mean of its models in float64, as `idun.gbm.predict_mean` does.

Several entries are blended by rank: each entry's predictions are ranked within the era (ties averaged) and scaled to
(0, 1], and the weighted mean of those ranks is ranked again; one entry's prediction is its own rank. The function's
``raw`` attribute, ``predict.raw(live_features)``, returns each entry's predictions before ranking. `save` writes the
function with cloudpickle under the Python version of Numerai's runtime: a function's bytecode does not carry across
Python versions.

The live features must hold every feature the models read (other columns are ignored): each obfuscated value exactly
one of 0..4 and every country one the training data has. Anything else raises ``ValueError`` rather than predict on
inputs the models never saw.
"""

import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.data import PackedDataset
from idun.gbm import Booster
from idun.jarf import JARFModel, direction_names
from idun.realmlp import RealMLPModel
from idun.tabm import TabMModel, hl_gauss
from idun.tabm.directions import PROJECT_ROWS

RUNTIME_PYTHON = (3, 12)  # numerai-predict's image (docs.numer.ai, "Model Uploads")
CHUNK = 512  # rows per forward pass of a network: bounds its memory, not its result
MISSING_BIN = 255  # a named feature's NaN (idun.data.layout)
CODE_LEVELS = 5  # obfuscated values are exactly 0..4
SELU_ALPHA = 1.6732632423543772848170429916717  # idun.realmlp.reference
SELU_SCALE = 1.0507009873554804934193349852946
_HEAP = ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value")
_NODES = (*_HEAP[:-1], "left", "right", "leaf_value")

Model = Booster | JARFModel | TabMModel | RealMLPModel
Spec = dict[str, Any]
Predict = Callable[..., Any]


def predictor(entries: Sequence[Model | Sequence[Model]], data: PackedDataset, *,
              weights: Sequence[float] | None = None) -> Predict:  # fmt: skip
    """Return Numerai's ``predict(live_features, live_benchmark_models)`` for ``entries``.

    Each entry is a model or a bag (a non-empty list of models of one kind), all trained on ``data`` (the packed
    dataset whose features, countries and input layout the live features are mapped onto). ``weights``: one positive
    weight per entry for the rank blend (equal by default). Raises ``ValueError`` for a model trained on other features,
    a multi-target network, a mixed bag, or weights that do not fit.
    """
    if not entries:
        raise ValueError("predictor needs at least one model")
    blend = [1.0] * len(entries) if weights is None else [float(w) for w in weights]
    if len(blend) != len(entries) or not all(w > 0 and math.isfinite(w) for w in blend):
        raise ValueError(f"weights must be {len(entries)} positive finite numbers, got {weights!r}")
    layout = _layout(data)
    specs = [_entry(entry, layout) for entry in entries]

    def raw(live_features: Any) -> list[NDArray[np.float32]]:
        return _raw(specs, layout, live_features)

    def predict(live_features: Any, live_benchmark_models: Any = None) -> Any:  # noqa: ARG001 - Numerai's signature
        ranked = _blend(_raw(specs, layout, live_features), blend)
        return type(live_features)({"prediction": ranked}, index=live_features.index)  # a DataFrame, as its input

    predict.raw = raw  # type: ignore[attr-defined]
    return predict


def save(predict: Predict, path: str | Path, *, python: tuple[int, int] = RUNTIME_PYTHON) -> Path:
    """Write ``predict`` to ``path`` with cloudpickle, this module by value; return the path.

    ``python``: the runtime's Python version; ``RuntimeError`` when this interpreter is another, since the pickle
    carries bytecode (run the export under Python 3.12 for numerai-predict's image).
    """
    import cloudpickle  # type: ignore[import-untyped]  # noqa: PLC0415 - only a save needs it; it has no annotations

    if sys.version_info[:2] != tuple(python):
        raise RuntimeError(f"this is Python {sys.version_info[0]}.{sys.version_info[1]}; the runtime runs "
                           f"{python[0]}.{python[1]}, and a pickled function's bytecode does not carry across "
                           "versions: save under that version")  # fmt: skip
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        payload = cloudpickle.dumps(predict)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
    out = Path(path)
    out.write_bytes(payload)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Export: models to arrays (runs where idun is installed).
# ---------------------------------------------------------------------------------------------------------------------
def _layout(data: PackedDataset) -> Spec:
    named, obfuscated = list(data.named_features), list(data.obfuscated_features)
    if len(data.categories) != 1:
        raise ValueError("the upload reads Numerai's one categorical column (the country)")
    return {"named": named, "obfuscated": obfuscated, "categorical": data.feature_names[len(named) + len(obfuscated)],
            "levels": list(data.categories[0])}  # fmt: skip


def _n_inputs(layout: Spec) -> int:
    return len(layout["named"]) + len(layout["obfuscated"]) + len(layout["levels"])


def _entry(entry: Model | Sequence[Model], layout: Spec) -> Spec:
    models = list(entry) if isinstance(entry, Sequence) else [entry]
    if not models or len({type(m) for m in models}) != 1:
        raise ValueError("a bag is a non-empty list of models of one kind")
    return {"kind": type(models[0]).__name__, "models": [_model(m, layout) for m in models]}


def _model(model: Model, layout: Spec) -> Spec:
    if isinstance(model, JARFModel):
        kept = list(layout["obfuscated"]) if model.kept is None else list(model.kept)
        if unknown := [name for name in kept if name not in layout["obfuscated"]]:
            raise ValueError(f"the JARF model keeps obfuscated features the dataset has not: {unknown[:3]}")
        names = direction_names(model.directions.k, model.prefix)
        features = (*layout["named"], *names, *kept, layout["categorical"])
        if model.booster.feature_names != features or model.booster.categories != (tuple(layout["levels"]),):
            raise ValueError("the JARF model was trained on other features, kept features or countries than the "
                             "dataset's")  # fmt: skip
        if model.directions.n_inputs != _n_inputs(layout):
            raise ValueError(f"the directions are over {model.directions.n_inputs} inputs; the dataset gives "
                             f"{_n_inputs(layout)}")  # fmt: skip
        index = np.array([layout["obfuscated"].index(name) for name in kept], np.int64)
        return {**_trees(model.booster), "vectors": np.asarray(model.directions.vectors, np.float32), "kept": index}
    if isinstance(model, Booster):
        features = (*layout["named"], *layout["obfuscated"], layout["categorical"])
        if model.feature_names != features or model.categories != (tuple(layout["levels"]),):
            raise ValueError("the idun_gbm model was trained on other features or countries than the dataset's")
        return _trees(model)
    if not isinstance(model, (TabMModel, RealMLPModel)):
        raise TypeError(f"{type(model).__name__} is not an idun_gbm, idun_tabm or idun_realmlp model")
    if model.multi_target:
        raise ValueError("an upload predicts one target: a multi-target network is not supported")
    if isinstance(model, TabMModel) and model.sequence_inputs:
        raise ValueError("an upload predicts from each era's features alone: a TabM with a temporal encoder, which "
                         "reads each row's history, is not supported")  # fmt: skip
    if model.n_inputs != _n_inputs(layout):
        raise ValueError(f"the network reads {model.n_inputs} inputs; the dataset gives {_n_inputs(layout)}")
    if isinstance(model, TabMModel):
        t, p = model.weights, model.params
        values = hl_gauss.centres(p.n_bins, p.hl_low, p.hl_high) if p.objective == "hl_gauss" else None
        return {"objective": p.objective, "weight": t.weight, "r": t.r, "s": t.s, "bias": t.bias, "head": t.head,
                "head_bias": t.head_bias, "values": values}  # fmt: skip
    w = model.weights
    return {"objective": model.params.objective, "clip": float(model.params.clip), "median": model.preprocess.median,
            "factor": model.preprocess.factor, "w1": w.w1, "b1": w.b1, "w2": w.w2, "b2": w.b2, "scale": w.scale,
            "weight": w.weight, "bias": w.bias, "act": w.act, "out": w.out, "out_bias": w.out_bias}  # fmt: skip


def _trees(model: Booster) -> Spec:
    """Return the forest as arrays: the named features' edges, the growth and the node fields."""
    names = _NODES if model.params.leafwise else _HEAP
    return {"edges": np.asarray(model.named_edges, np.float32), "leafwise": model.params.leafwise,
            **{name: np.asarray(getattr(model.forest, name)) for name in names}}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# Runtime: what runs where Numerai predicts. NumPy only, and nothing from idun.
# ---------------------------------------------------------------------------------------------------------------------
def _raw(specs: list[Spec], layout: Spec, frame: Any) -> list[NDArray[np.float32]]:
    named, codes, country = _columns(layout, frame)
    out = []
    for spec in specs:
        each = {"Booster": _gbm, "JARFModel": _jarf, "TabMModel": _tabm, "RealMLPModel": _realmlp}[spec["kind"]]
        total = np.zeros(country.size, np.float64)
        for model in spec["models"]:
            total += each(model, layout, named, codes, country)
        out.append((total / len(spec["models"])).astype(np.float32))
    return out


def _columns(layout: Spec, frame: Any) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.int64]]:
    """Return the named values ``(n, named)`` (NaN kept), the codes ``(n, obfuscated)`` and the country codes."""
    wanted = [*layout["named"], *layout["obfuscated"], layout["categorical"]]
    if missing := [name for name in wanted if name not in frame.columns]:
        raise ValueError(f"live_features lacks {len(missing)} of the models' features, first {missing[:3]}")
    named = np.asarray(frame[layout["named"]].to_numpy(dtype=np.float32, na_value=np.nan), np.float32)
    values = np.asarray(frame[layout["obfuscated"]].to_numpy(dtype=np.float32, na_value=np.nan), np.float32)
    with np.errstate(invalid="ignore"):
        codes = values.astype(np.uint8)
    if not (codes.max(initial=0) < CODE_LEVELS and np.array_equal(codes, values)):  # NaN never compares equal
        row, feature = (int(i) for i in np.argwhere((codes >= CODE_LEVELS) | (codes != values))[0])
        raise ValueError(f"{layout['obfuscated'][feature]} = {values[row, feature]} in row {row}, not one of 0..4")
    raw = frame[layout["categorical"]]
    if bool(raw.isna().any()):
        raise ValueError(f"{layout['categorical']} has missing values")
    index = {level: i for i, level in enumerate(layout["levels"])}
    labels = [str(v) for v in raw.to_numpy(dtype=object)]
    if unseen := sorted(set(labels) - set(index)):
        raise ValueError(f"{layout['categorical']} has levels the training data has not: {unseen}")
    return named, codes, np.array([index[v] for v in labels], np.int64)


def _gbm(spec: Spec, layout: Spec, named: NDArray[np.float32], codes: NDArray[np.uint8],
         country: NDArray[np.int64]) -> NDArray[np.float32]:  # fmt: skip
    """Walk the forest over the binned features (the packed dataset's bins); float32 sums in tree order."""
    del layout
    bins = np.empty((named.shape[1] + codes.shape[1] + 1, country.size), np.uint8)
    for j in range(named.shape[1]):
        column = np.searchsorted(spec["edges"][j], named[:, j], side="left").astype(np.uint8)
        column[np.isnan(named[:, j])] = MISSING_BIN
        bins[j] = column
    bins[named.shape[1] : -1] = codes.T
    bins[-1] = country.astype(np.uint8)
    rows = np.arange(country.size)
    leaf_value = spec["leaf_value"]
    internal = leaf_value.shape[1] - 1
    acc = np.zeros(country.size, np.float32)
    for t in range(leaf_value.shape[0]):
        if spec["leafwise"]:
            leaf = _walk_nodes(spec, t, bins, rows, internal)
        else:
            h = np.zeros(country.size, np.int64)
            for _ in range(internal.bit_length()):  # the depth: 2^depth − 1 internal nodes
                h = 2 * h + 1 + _goes_right(spec, t, h, bins, rows)
            leaf = h - internal
        acc += leaf_value[t, leaf]
    return acc


def _jarf(spec: Spec, layout: Spec, named: NDArray[np.float32], codes: NDArray[np.uint8],
          country: NDArray[np.int64]) -> NDArray[np.float32]:  # fmt: skip
    """Project the networks' inputs on the stored directions, append them to the named values, walk the trees."""
    dense = _dense(layout, named, codes, country, raw_codes=False)
    projections = np.empty((dense.shape[0], spec["vectors"].shape[1]), np.float32)
    for start in range(0, dense.shape[0], PROJECT_ROWS):  # one matmul per chunk, as idun projects
        projections[start : start + PROJECT_ROWS] = dense[start : start + PROJECT_ROWS] @ spec["vectors"]
    return _gbm(spec, layout, np.concatenate([named, projections], axis=1), codes[:, spec["kept"]], country)


def _walk_nodes(spec: Spec, t: int, bins: NDArray[np.uint8], rows: NDArray[np.int64],
                internal: int) -> NDArray[np.int64]:  # fmt: skip
    """Return each row's leaf in leaf-wise tree ``t``: child ``c >= 0`` is a node, ``c < 0`` leaf ``~c``."""
    node = np.zeros(rows.size, np.int64)
    child = np.full(rows.size, -1, np.int64)
    done = np.zeros(rows.size, bool)
    for _ in range(internal):  # every child node comes after its parent: a leaf within `internal` steps
        right = _goes_right(spec, t, node, bins, rows) == 1
        step = np.where(right, spec["right"][t, node], spec["left"][t, node]).astype(np.int64)
        ended = ~done & (step < 0)
        child[ended] = step[ended]
        done |= ended
        if done.all():
            break
        node = np.where(done, node, step)
    return ~child


def _goes_right(spec: Spec, t: int, node: NDArray[np.int64], bins: NDArray[np.uint8],
                rows: NDArray[np.int64]) -> NDArray[np.int64]:  # fmt: skip
    """Return 1 where a row goes right at its node of tree ``t``, else 0 (idun_gbm's split rule)."""
    kind = spec["kind"][t, node]
    b = bins[spec["feature"][t, node], rows]
    numeric = np.where(b == MISSING_BIN, spec["missing_left"][t, node] == 0, b > spec["threshold"][t, node])
    shift = np.where(kind == 2, b, 0).astype(np.uint64)  # a categorical bin is a category code < 64
    outside = ((spec["cat_left"][t, node] >> shift) & np.uint64(1)) == 0  # listed categories go left
    return np.where(kind == 0, 0, np.where(kind == 2, outside, numeric)).astype(np.int64)  # pass-through: left


def _dense(layout: Spec, named: NDArray[np.float32], codes: NDArray[np.uint8], country: NDArray[np.int64], *,
           raw_codes: bool) -> NDArray[np.float32]:  # fmt: skip
    """Return the networks' inputs as `idun.nn.inputs.dense_inputs` builds them."""
    values = np.nan_to_num(named, nan=0.0)
    obfuscated = codes.astype(np.float32) if raw_codes else (codes.astype(np.float32) - 2.0) / 2.0
    one_hot = np.zeros((country.size, len(layout["levels"])), np.float32)
    one_hot[np.arange(country.size), country] = 1.0
    return np.concatenate([values, obfuscated, one_hot], axis=1)


def _expected(logits: NDArray[np.float64], *, members: bool,
              values: NDArray[np.float64] | None = None) -> NDArray[np.float64]:  # fmt: skip
    """Return the expected class of the member-averaged softmax (TabM), or of the softmax (RealMLP).

    Given ``values`` (HL-Gauss's bin centres), the expected value of the outputs instead of the expected class.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    classes = np.arange(logits.shape[-1], dtype=np.float64) if values is None else values
    expected: NDArray[np.float64] = (p.mean(axis=1) * classes).sum(axis=-1) if members else (p * classes).sum(-1)
    return expected


def _tabm(spec: Spec, layout: Spec, named: NDArray[np.float32], codes: NDArray[np.uint8],
          country: NDArray[np.int64]) -> NDArray[np.float32]:  # fmt: skip
    """TabM's inference in float64, as `idun.tabm.reference.forward` and `predict`."""
    x32 = _dense(layout, named, codes, country, raw_codes=False)
    weight, r, s, bias = ([a.astype(np.float64) for a in spec[key]] for key in ("weight", "r", "s", "bias"))
    head, head_bias = spec["head"].astype(np.float64), spec["head_bias"].astype(np.float64)
    out = np.empty(country.size, np.float32)
    for lo in range(0, country.size, CHUNK):
        x = x32[lo : lo + CHUNK].astype(np.float64)
        h = np.broadcast_to(x[:, None, :], (x.shape[0], r[0].shape[0], x.shape[1]))
        for i in range(len(weight)):
            z = (h * r[i]) @ weight[i].T * s[i] + bias[i]
            h = np.maximum(z, 0) * np.ones_like(z)
        logits = np.einsum("bjw,jwc->bjc", h, head) + head_bias
        pred = (logits.mean(axis=1)[:, 0] if spec["objective"] == "mse"
                else _expected(logits, members=True, values=spec["values"]))  # fmt: skip
        out[lo : lo + CHUNK] = pred
    return out


def _selu(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return SELU_SCALE * np.where(v > 0, v, SELU_ALPHA * np.expm1(np.minimum(v, 0.0)))


def _realmlp(spec: Spec, layout: Spec, named: NDArray[np.float32], codes: NDArray[np.uint8],
             country: NDArray[np.int64]) -> NDArray[np.float32]:  # fmt: skip
    """RealMLP's inference in float64, as `idun.realmlp.reference.preprocess`, `forward` and `predict`."""
    raw = _dense(layout, named, codes, country, raw_codes=True).astype(np.float64)
    v = (raw - spec["median"]) * spec["factor"]
    c32 = (v / np.sqrt(1.0 + v * v / (spec["clip"] * spec["clip"]))).astype(np.float32)
    w1, b1, w2, b2, scale, out_w, out_b = (spec[k].astype(np.float64) for k in ("w1", "b1", "w2", "b2", "scale", "out",
                                                                                "out_bias"))  # fmt: skip
    weight, bias, act = ([a.astype(np.float64) for a in spec[key]] for key in ("weight", "bias", "act"))
    out = np.empty(country.size, np.float32)
    for lo in range(0, country.size, CHUNK):
        c = c32[lo : lo + CHUNK].astype(np.float64)
        arg = 2.0 * math.pi * c[:, :, None] * w1 + b1
        e = np.einsum("bfq,fqe->bfe", np.cos(arg), w2) + b2
        h = np.concatenate([e.reshape(len(c), -1), c], axis=1) * scale
        for i, w in enumerate(weight):
            pre = h @ w / math.sqrt(w.shape[0]) + bias[i]
            a = pre + (_selu(pre) - pre) * act[i]
            h = a * np.ones_like(a)
        logits = h @ out_w / math.sqrt(out_w.shape[0]) + out_b
        out[lo : lo + CHUNK] = logits[:, 0] if spec["objective"] == "mse" else _expected(logits, members=False)
    return out


def _rank(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return each value's rank scaled to (0, 1]: average ranks of ties, over the count (pandas' rank(pct=True))."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], ordered.size]
    average = (starts + ends + 1) / 2.0  # ranks start + 1 .. end, averaged
    ranks = np.empty(values.size, np.float64)
    ranks[order] = np.repeat(average, ends - starts)
    return ranks / values.size


def _blend(predictions: list[NDArray[np.float32]], weights: list[float]) -> NDArray[np.float64]:
    if len(predictions) == 1:
        return _rank(predictions[0])
    mean = sum(w * _rank(p) for w, p in zip(weights, predictions, strict=True)) / sum(weights)
    return _rank(np.asarray(mean, np.float64))


__all__ = ["RUNTIME_PYTHON", "predictor", "save"]
