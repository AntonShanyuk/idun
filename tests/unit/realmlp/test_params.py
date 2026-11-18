import pytest

from idun.realmlp import RealMLPParams


def test_defaults_are_pytabkit_realmlp_td_settings() -> None:
    p = RealMLPParams()
    assert (p.blocks, p.hidden, p.n_classes, p.frequencies, p.embed) == (3, 256, 2, 16, 3)
    assert (p.learning_rate, p.weight_decay, p.beta1, p.beta2, p.eps) == (0.04, 0.02, 0.9, 0.95, 1e-8)
    assert (p.dropout, p.label_smoothing, p.clip, p.embed_sigma) == (0.15, 0.1, 3.0, 0.1)
    assert (p.scale_lr_factor, p.bias_lr_factor, p.act_lr_factor, p.embed_lr_factor) == (6.0, 0.1, 0.1, 0.1)
    assert (p.batch_size, p.epochs, p.init_rows, p.seed) == (256, 256, 0, 0)  # 0: pytabkit's count for the table
    assert p.own_last_block is False  # every hidden block shared, as one network per target has them


@pytest.mark.parametrize(("field", "value"), [("hidden", 0), ("dropout", 1.0), ("label_smoothing", -0.1),
                                              ("learning_rate", 0.0), ("seed", -1), ("n_classes", 1),
                                              ("act", "gelu"), ("init_rows", -1)])  # fmt: skip
def test_bad_values_raise(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"out of range: {field}="):
        RealMLPParams(**{field: value})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wrong type"):
        RealMLPParams(hidden=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="own_last_block has the wrong type"):
        RealMLPParams(own_last_block=1)  # type: ignore[arg-type]


def test_the_objective_fixes_the_class_count() -> None:
    assert RealMLPParams(n_classes=1, objective="mse").objective == "mse"
    with pytest.raises(ValueError, match="n_classes"):
        RealMLPParams(n_classes=3, objective="mse")
    with pytest.raises(ValueError, match="objective"):
        RealMLPParams(objective="hinge")
    with pytest.raises(ValueError, match="wrong type"):
        RealMLPParams(objective=1)  # type: ignore[arg-type]


def test_hidden_sizes_set_each_layers_width() -> None:
    assert RealMLPParams().widths == (256, 256, 256)
    assert RealMLPParams(hidden_sizes=[512, 64]).hidden_sizes == (512, 64)  # type: ignore[arg-type]  # a list, as JSON gives it
    with pytest.raises(ValueError, match="one or the other"):
        RealMLPParams(hidden_sizes=(64,), blocks=2)
    with pytest.raises(ValueError, match="out of range: hidden_sizes"):
        RealMLPParams(hidden_sizes=(64, 0))
