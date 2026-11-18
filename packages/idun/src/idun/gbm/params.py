"""Training parameters: one frozen, validated dataclass (docs/api/python-api.md).

The defaults are neutral library defaults.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Self

CATEGORICAL_METHODS = ("sorted", "one_vs_rest")
GROWTH_POLICIES = ("depthwise", "leafwise")
MAX_DEPTH = 8  # depth-wise: node ids are uint8, 2^8 nodes on the deepest level
MAX_LEAVES = 256  # leaf-wise: node ids are uint8 leaf ids
NO_DEPTH_LIMIT = -1  # leaf-wise max_depth: no limit (LightGBM's -1)
U32_MAX = 0xFFFF_FFFF


@dataclass(frozen=True, slots=True)
class Params:
    """Gradient boosting parameters. Every field is checked at construction; a bad value raises ``ValueError``.

    The categorical fields mirror LightGBM's (``cat_smooth``, ``cat_l2``, ``max_cat_threshold``,
    ``min_data_per_group``), with LightGBM's defaults; ``categorical`` picks the split method.

    The regularisers are LightGBM's: ``lambda_l1`` shrinks a leaf's gradient sum towards 0 (``ThresholdL1``),
    ``min_gain_to_split`` is the gain a split must exceed, ``min_sum_hessian_in_leaf`` the hessian sum each side needs
    (the split search's hessians are the in-bag row counts), ``max_delta_step`` > 0 bounds a leaf's output before the
    learning rate (0: none). Their defaults turn them off.

    ``second_order``: the split search reads the hessians of a `NewtonObjective`, of row weights and of a
    multi-class softmax (LightGBM's rule, ``G² / (H + λ)`` with ``H`` their sum; ``min_sum_hessian_in_leaf`` sums them).
    False, the default, keeps first-order splits for every objective (the row count as the hessian); the
    leaves are Newton steps either way.

    ``extra_trees`` and ``feature_fraction_bynode`` are LightGBM's: each node tries one random threshold per
    numeric feature (categorical features keep their search), and searches a random fraction of the tree's sampled
    features. Both are drawn per node from the seed and the tree's index on the host, so every backend fits the same
    trees; off (False, 1.0) by default.

    ``growth``: ``"depthwise"`` grows each tree level by level to ``max_depth`` (1..8), keeping the
    highest-gain splits of a level when it would exceed ``max_leaves`` (at most ``2^max_depth``). ``"leafwise"`` is
    LightGBM's best-first growth: it splits the leaf with the highest gain until the tree has ``max_leaves`` leaves
    (at most 256) or no leaf has a split; ``max_depth`` is then ``-1`` (no limit) or a depth no leaf may exceed.
    """

    num_trees: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    max_leaves: int = 64
    growth: Literal["depthwise", "leafwise"] = "depthwise"
    min_data_in_leaf: int = 20
    lambda_l2: float = 0.0
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    grad_quant_bins: int = 4
    categorical: Literal["sorted", "one_vs_rest"] = "sorted"
    cat_smooth: float = 10.0
    cat_l2: float = 10.0
    max_cat_threshold: int = 32
    min_data_per_group: int = 100
    seed: int = 0
    lambda_l1: float = 0.0
    min_gain_to_split: float = 0.0
    min_sum_hessian_in_leaf: float = 1e-3
    max_delta_step: float = 0.0
    second_order: bool = False
    extra_trees: bool = False
    feature_fraction_bynode: float = 1.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif field.type is float:
                ok = isinstance(value, int | float) and not isinstance(value, bool)
            elif field.type is bool:
                ok = isinstance(value, bool)
            else:
                ok = isinstance(value, str)
            if not ok:
                raise ValueError(f"Params.{field.name} has the wrong type: {value!r}")
        if self.growth == "leafwise":
            depth_ok = self.max_depth == NO_DEPTH_LIMIT or self.max_depth >= 1
            limit = MAX_LEAVES if self.max_depth == NO_DEPTH_LIMIT else 2 ** min(self.max_depth, 16)
            most_leaves = min(MAX_LEAVES, limit)
        else:
            depth_ok = 1 <= self.max_depth <= MAX_DEPTH
            most_leaves = 2**self.max_depth
        checks = {
            "growth": self.growth in GROWTH_POLICIES,
            "num_trees": self.num_trees >= 1,
            "learning_rate": self.learning_rate > 0.0,
            "max_depth": depth_ok,
            "max_leaves": 2 <= self.max_leaves <= most_leaves,
            "min_data_in_leaf": self.min_data_in_leaf >= 1,
            "lambda_l2": self.lambda_l2 >= 0.0,
            "feature_fraction": 0.0 < self.feature_fraction <= 1.0,
            "bagging_fraction": 0.0 < self.bagging_fraction <= 1.0,
            "bagging_freq": self.bagging_freq >= 0,
            "grad_quant_bins": self.grad_quant_bins in (2, 4, 8, 16, 32, 64, 128),
            "categorical": self.categorical in CATEGORICAL_METHODS,
            "cat_smooth": self.cat_smooth >= 0.0,
            "cat_l2": self.cat_l2 >= 0.0,
            "max_cat_threshold": self.max_cat_threshold >= 1,
            "min_data_per_group": self.min_data_per_group >= 1,
            "seed": 0 <= self.seed <= U32_MAX,
            "lambda_l1": self.lambda_l1 >= 0.0,
            "min_gain_to_split": self.min_gain_to_split >= 0.0,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf >= 0.0,
            "max_delta_step": self.max_delta_step >= 0.0,
            "feature_fraction_bynode": 0.0 < self.feature_fraction_bynode <= 1.0,
        }
        if failed := [name for name, ok in checks.items() if not ok]:
            raise ValueError(f"invalid Params.{failed[0]} = {getattr(self, failed[0])!r}")

    @property
    def leafwise(self) -> bool:
        """Whether trees grow best-first (``growth="leafwise"``)."""
        return self.growth == "leafwise"

    @property
    def bagging(self) -> bool:
        """Whether rows are bagged (LightGBM bags only when ``bagging_fraction < 1`` and ``bagging_freq > 0``)."""
        return self.bagging_fraction < 1.0 and self.bagging_freq > 0

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a JSON-ready dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        """Rebuild from `to_dict` output; unknown keys raise ``ValueError``."""
        known = {field.name for field in dataclasses.fields(cls)}
        if unknown := sorted(set(values) - known):
            raise ValueError(f"unknown Params fields {unknown}")
        return cls(**values)
