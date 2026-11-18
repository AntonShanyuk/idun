"""RealMLP training parameters: one frozen, validated dataclass.

The defaults are pytabkit 1.7.3's ``RealMLP_TD_Classifier`` defaults (256 epochs, 2 classes).
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

U32_MAX = 0xFFFF_FFFF
OBJECTIVES = ("cross_entropy", "mse")  # classification over n_classes; regression (n_classes 1)
ACTIVATIONS = ("selu", "mish", "relu")  # the hidden layers' activation, pytabkit's names


@dataclass(frozen=True, slots=True)
class RealMLPParams:
    """RealMLP-TD's settings: pytabkit 1.7.3 ``RealMLP_TD_Classifier`` defaults.

    The spec is docs/architecture/networks.md §8. Every field is checked at construction; a bad value raises
    ``ValueError``.

    ``frequencies`` cosines per input feeding ``embed`` values (PBLD), ``blocks`` hidden layers of ``hidden`` units with
    a parametric activation (``act``: SELU by default, the classifier's; ``"mish"``, pytabkit's regressor's, or
    ``"relu"``) and dropout, a linear output onto ``n_classes``. ``hidden_sizes`` (pytabkit's) gives each
    hidden layer's width instead of ``blocks`` × ``hidden`` (the default, empty, keeps them; not both). Learning-rate
    factors multiply ``learning_rate`` per parameter kind; biases are not decayed. ``init_rows`` rows of the training
    data fit the preprocessing and the data-dependent initialisation; 0 (the default) takes pytabkit's own count for
    the table, `train.pytabkit_init_rows`.

    ``own_last_block``: experimental, not covered by the documentation.

    ``embedding_size`` (pytabkit's): the learned values of each categorical input that `train_realmlp` embeds
    (its ``categories``) instead of reading it one-hot; pytabkit's 8.
    """

    blocks: int = 3
    hidden: int = 256
    n_classes: int = 2
    frequencies: int = 16
    embed: int = 3
    embed_sigma: float = 0.1
    clip: float = 3.0
    dropout: float = 0.15
    label_smoothing: float = 0.1
    learning_rate: float = 0.04
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    scale_lr_factor: float = 6.0
    bias_lr_factor: float = 0.1
    act_lr_factor: float = 0.1
    embed_lr_factor: float = 0.1
    batch_size: int = 256
    epochs: int = 256
    init_rows: int = 0
    seed: int = 0
    own_last_block: bool = False
    objective: str = "cross_entropy"
    lr_schedule: str = "coslog4"
    dropout_schedule: str = "flat_cos"
    wd_schedule: str = "flat_cos"
    act: str = "selu"
    hidden_sizes: tuple[int, ...] = ()
    embedding_size: int = 8

    def __post_init__(self) -> None:
        widths: Any = self.hidden_sizes
        if isinstance(widths, list):  # a list from JSON or a caller: the same widths
            object.__setattr__(self, "hidden_sizes", tuple(widths))
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "hidden_sizes":
                ok = isinstance(value, tuple) and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            elif field.type is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif field.type is bool:
                ok = isinstance(value, bool)
            elif field.type is str:
                ok = isinstance(value, str)
            else:
                ok = isinstance(value, int | float) and not isinstance(value, bool)
            if not ok:
                raise ValueError(f"RealMLPParams.{field.name} has the wrong type: {value!r}")
        checks = {
            **{name: getattr(self, name) >= 1 for name in ("blocks", "hidden", "frequencies", "embed", "batch_size",
                                                          "epochs", "embedding_size")},
            "init_rows": self.init_rows >= 0,
            "n_classes": self.n_classes >= 2 if self.objective == "cross_entropy" else self.n_classes == 1,
            "objective": self.objective in OBJECTIVES,
            "act": self.act in ACTIVATIONS,
            **{name: getattr(self, name) > 0.0 for name in ("embed_sigma", "clip", "learning_rate", "eps",
                                                           "scale_lr_factor", "bias_lr_factor", "act_lr_factor",
                                                           "embed_lr_factor")},
            "dropout": 0.0 <= self.dropout < 1.0,
            "label_smoothing": 0.0 <= self.label_smoothing < 1.0,
            "weight_decay": self.weight_decay >= 0.0,
            "beta1": 0.0 <= self.beta1 < 1.0,
            "beta2": 0.0 <= self.beta2 < 1.0,
            "seed": 0 <= self.seed <= U32_MAX,
            "lr_schedule": self.lr_schedule in ("coslog4", "cosine", "constant"),
            "dropout_schedule": self.dropout_schedule in ("flat_cos", "constant"),
            "wd_schedule": self.wd_schedule in ("flat_cos", "constant"),
            "hidden_sizes": all(width >= 1 for width in self.hidden_sizes),
        }  # fmt: skip
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"RealMLPParams out of range: {', '.join(f'{n}={getattr(self, n)!r}' for n in bad)}")
        if self.hidden_sizes and (self.blocks, self.hidden) != (3, 256):
            raise ValueError("hidden_sizes or blocks and hidden: the hidden layers' widths are one or the other")

    @property
    def widths(self) -> tuple[int, ...]:
        """Return the hidden layers' widths: ``hidden_sizes``, or ``blocks`` layers of ``hidden`` units."""
        return self.hidden_sizes or (self.hidden,) * self.blocks
