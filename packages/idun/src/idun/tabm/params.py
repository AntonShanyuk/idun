"""TabM training parameters: one frozen, validated dataclass.

The defaults are the `tabm` package's (0.0.3) `TabM.make` defaults (32 members, 3 blocks of 512, dropout 0.1) and the
paper's training: AdamW at 2e-3 with weight decay 3e-4, 50 epochs of 256-row batches, 2 classes.
"""

import dataclasses
import math
from dataclasses import dataclass

U32_MAX = 0xFFFF_FFFF
STORAGES = ("fp32", "bf16")
OBJECTIVES = ("cross_entropy", "mse", "hl_gauss")  # classification over n_classes; regression (n_classes 1), two ways
MAX_BINS = 512  # HL-Gauss: the Metal head runs its bins in groups of 8
ENCODERS = ("none", "gru")  # the temporal encoder of per-row sequences (`idun.tabm.encoder`)
MAX_ENCODER_WIDTH = 128  # encoder_embed, encoder_hidden: Metal keeps a row's encoder state in threadgroup memory

NUM_EMB_TYPES = ("none", "pwl")  # numeric embeddings: none, or piecewise-linear (`idun.tabm.embeddings`)
MAX_EMB_BINS = 256  # num_emb_n_bins
MAX_D_EMBEDDING = 128  # d_embedding
LR_SCHEDULES = ("constant", "cosine", "linear")  # the learning rate over the steps (`scheduled_rate`)


@dataclass(frozen=True, slots=True)
class TabMParams:
    """BatchEnsemble MLP and AdamW settings. Every field is checked at construction; a bad value raises ``ValueError``.

    ``members`` (k) networks share each block's weight matrix and differ by rank-1 adapters (``r`` on the input, ``s``
    on the output) and their own biases; block 0's ``r`` starts as random signs. Each block is Linear → ReLU → Dropout.
    The head is one linear layer per member onto `outputs` values. AdamW decays every parameter (one parameter group, as
    ``torch.optim.AdamW(model.parameters())``). ``storage`` is how the Metal backend stores activations and their
    gradients: ``"fp32"`` (the default) or ``"bf16"`` (half the memory traffic; weights, AdamW moments and all
    arithmetic stay fp32). ``micro_batch`` between 1 and ``batch_size`` − 1 makes the Metal backend run each batch in
    micro-batches of that many rows (the last may be smaller) with its activation buffers sized for one micro-batch, the
    gradients summed over them in a fixed order before the one AdamW step: the same update as the whole batch up to
    float summation order, for less memory. 0 (the default) runs the whole batch at once; ``batch_size`` or more raises,
    since it would split nothing. The reference backend always computes in float64 on the whole batch.

    ``objective``: ``"cross_entropy"`` (classification over ``n_classes``), ``"mse"`` (regression, ``n_classes`` 1, one
    output) or ``"hl_gauss"`` (regression as classification over ``n_bins`` uniform bins of ``[hl_low, hl_high]`` with
    labels smoothed by N(target, ``hl_sigma``); ``n_classes`` 1; `idun.tabm.hl_gauss`). The defaults suit a standardized
    target: 64 bins on [−4, 4] and a sigma of 0.75 bin widths, 0.09375.

    ``encoder``: ``"none"`` (the default) is TabM on the snapshot inputs alone. ``"gru"`` adds a temporal encoder that
    reads a sequence per row (`idun.tabm.Sequences`): each step's inputs through a shared Linear to ``encoder_embed``
    units and a GELU, then a GRU of ``encoder_hidden`` units over the steps, oldest first; its last hidden state is
    prepended to the row's inputs, and the encoder trains with the network (`idun.tabm.encoder`).

    ``num_emb_type``: ``"none"`` (the default) feeds the inputs to block 0 as they are. ``"pwl"`` gives each embedded
    input column a piecewise-linear embedding of ``d_embedding`` values over at most ``num_emb_n_bins`` quantile bins of
    its training values (rtdl's version "B", as pytabkit's TabM: `idun.tabm.embeddings`), which block 0 reads in its
    place; block 0's ``r`` then starts N(0, 1), one draw per member and column shared by the column's ``d_embedding``
    values, as pytabkit's. Not with an encoder or ``"bf16"`` storage.

    ``gradient_clipping_norm`` (pytabkit's): above 0, each step's gradients are scaled as
    ``torch.nn.utils.clip_grad_norm_`` scales them, by min(1, norm / (‖g‖ + 1e-6)) over every parameter, before AdamW;
    0 (the default) leaves them as they are, bit for bit.
    """

    members: int = 32
    blocks: int = 3
    width: int = 512
    dropout: float = 0.1
    n_classes: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    batch_size: int = 256
    epochs: int = 50
    seed: int = 0
    storage: str = "fp32"
    micro_batch: int = 0
    objective: str = "cross_entropy"
    n_bins: int = 64
    hl_low: float = -4.0
    hl_high: float = 4.0
    hl_sigma: float = 0.09375
    encoder: str = "none"
    encoder_embed: int = 32
    encoder_hidden: int = 64
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    num_emb_type: str = "none"
    num_emb_n_bins: int = 48
    d_embedding: int = 16
    gradient_clipping_norm: float = 0.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif field.type is str:
                ok = isinstance(value, str)
            else:
                ok = isinstance(value, int | float) and not isinstance(value, bool)
            if not ok:
                raise ValueError(f"TabMParams.{field.name} has the wrong type: {value!r}")
        checks = {
            "members": self.members >= 1,
            "blocks": self.blocks >= 1,
            "width": self.width >= 1,
            "dropout": 0.0 <= self.dropout < 1.0,
            "n_classes": self.n_classes >= 2 if self.objective == "cross_entropy" else self.n_classes == 1,
            "learning_rate": self.learning_rate > 0.0,
            "weight_decay": self.weight_decay >= 0.0,
            "beta1": 0.0 <= self.beta1 < 1.0,
            "beta2": 0.0 <= self.beta2 < 1.0,
            "eps": self.eps > 0.0,
            "batch_size": self.batch_size >= 1,
            "epochs": self.epochs >= 1,
            "seed": 0 <= self.seed <= U32_MAX,
            "storage": self.storage in STORAGES,
            "micro_batch": 0 <= self.micro_batch < self.batch_size,
            "objective": self.objective in OBJECTIVES,
            "n_bins": 8 <= self.n_bins <= MAX_BINS and self.n_bins % 8 == 0,
            "hl_low": math.isfinite(self.hl_low) and self.hl_low < self.hl_high,
            "hl_high": math.isfinite(self.hl_high),
            "hl_sigma": math.isfinite(self.hl_sigma) and self.hl_sigma > 0.0,
            "lr_schedule": self.lr_schedule in LR_SCHEDULES,
            "lr_warmup_steps": self.lr_warmup_steps >= 0,
            "encoder": self.encoder in ENCODERS,
            "encoder_embed": 1 <= self.encoder_embed <= MAX_ENCODER_WIDTH,
            "encoder_hidden": 1 <= self.encoder_hidden <= MAX_ENCODER_WIDTH,
            "num_emb_type": self.num_emb_type in NUM_EMB_TYPES,
            "num_emb_n_bins": 2 <= self.num_emb_n_bins <= MAX_EMB_BINS,
            "d_embedding": 1 <= self.d_embedding <= MAX_D_EMBEDDING,
            "gradient_clipping_norm": math.isfinite(self.gradient_clipping_norm) and self.gradient_clipping_norm >= 0.0,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            values = ", ".join(f"{n}={getattr(self, n)!r}" for n in bad)
            if "micro_batch" in bad:
                values += " (a micro-batch has fewer rows than batch_size; 0 runs the whole batch)"
            raise ValueError(f"TabMParams out of range: {values}")
        if self.num_emb_type != "none" and (self.encoder != "none" or self.storage != "fp32"):
            raise ValueError(f"num_emb_type {self.num_emb_type!r} takes no encoder and fp32 storage, got encoder "
                             f"{self.encoder!r} and storage {self.storage!r}")  # fmt: skip

    @property
    def encoded(self) -> int:
        """Return the inputs the encoder prepends to a row's: ``encoder_hidden``, or 0 without an encoder."""
        return 0 if self.encoder == "none" else self.encoder_hidden

    @property
    def outputs(self) -> int:
        """Return the head's outputs per target: ``n_classes``, 1 for ``"mse"``, ``n_bins`` for ``"hl_gauss"``."""
        return {"mse": 1, "hl_gauss": self.n_bins}.get(self.objective, self.n_classes)


def scheduled_rate(params: "TabMParams", step: int, total: int) -> float | None:
    """Return step ``step``'s learning rate (of ``total``), or None for the constant ``learning_rate`` itself.

    A linear warm-up over the first ``lr_warmup_steps`` steps (``(step + 1) / warm-up`` of the rate), then
    ``lr_schedule`` over the rest: ``"constant"``, ``"cosine"`` (``(1 + cos(π t)) / 2``) or ``"linear"`` (``1 − t``)
    with ``t`` the share of the steps after the warm-up already taken. Computed in float64 on the host.
    """
    if params.lr_schedule == "constant" and params.lr_warmup_steps == 0:
        return None
    base, warm = params.learning_rate, params.lr_warmup_steps
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    if params.lr_schedule == "cosine":
        return base * 0.5 * (1.0 + math.cos(math.pi * t))
    if params.lr_schedule == "linear":
        return base * (1.0 - t)
    return base
