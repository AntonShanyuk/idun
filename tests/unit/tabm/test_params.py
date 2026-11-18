import dataclasses

import pytest

from idun.tabm import TabMParams


def test_defaults_are_tabm_make_defaults_and_the_papers_training() -> None:
    p = TabMParams()
    assert (p.members, p.blocks, p.width, p.dropout, p.n_classes) == (32, 3, 512, 0.1, 2)
    assert (p.learning_rate, p.weight_decay, p.beta1, p.beta2, p.eps) == (2e-3, 3e-4, 0.9, 0.999, 1e-8)
    assert (p.batch_size, p.epochs, p.seed, p.storage, p.micro_batch) == (256, 50, 0, "fp32", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.width = 256  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [("members", 0), ("blocks", 0), ("width", 0), ("dropout", 1.0), ("dropout", -0.1), ("n_classes", 1),
     ("learning_rate", 0.0), ("weight_decay", -1.0), ("beta1", 1.0), ("beta2", -0.5), ("eps", 0.0),
     ("batch_size", 0), ("epochs", 0), ("seed", -1), ("seed", 2**32), ("micro_batch", -1), ("micro_batch", 4096),
     ("micro_batch", 4097), ("n_bins", 4), ("n_bins", 60), ("n_bins", 520), ("hl_low", 4.0), ("hl_high", float("inf")),
     ("hl_sigma", 0.0)],
)  # fmt: skip
def test_out_of_range_values_raise(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"out of range: {field}="):
        TabMParams(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(("field", "value"), [("members", 2.0), ("members", True), ("dropout", "0.1"),
                                              ("micro_batch", 1024.0)])  # fmt: skip
def test_wrong_types_raise(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"TabMParams.{field} has the wrong type"):
        TabMParams(**{field: value})  # type: ignore[arg-type]


def test_a_micro_batch_is_smaller_than_the_batch() -> None:
    assert TabMParams(batch_size=4096, micro_batch=1024).micro_batch == 1024
    assert TabMParams(batch_size=100, micro_batch=99).micro_batch == 99
    for micro in (100, 101):  # the whole batch or more would split nothing
        with pytest.raises(ValueError, match=rf"out of range: micro_batch={micro} \(a micro-batch has fewer rows"):
            TabMParams(batch_size=100, micro_batch=micro)


def test_the_objective_fixes_the_class_count() -> None:
    assert TabMParams(n_classes=1, objective="mse").objective == "mse"
    with pytest.raises(ValueError, match="n_classes"):
        TabMParams(n_classes=5, objective="mse")
    with pytest.raises(ValueError, match="n_classes"):
        TabMParams(n_classes=1)
    with pytest.raises(ValueError, match="objective"):
        TabMParams(objective="hinge")


def test_hl_gauss_is_regression_with_a_head_of_n_bins() -> None:
    p = TabMParams(n_classes=1, objective="hl_gauss")
    assert (p.n_bins, p.hl_low, p.hl_high, p.hl_sigma, p.outputs) == (64, -4.0, 4.0, 0.09375, 64)
    assert TabMParams(n_classes=1, objective="hl_gauss", n_bins=128).outputs == 128
    assert (TabMParams().outputs, TabMParams(n_classes=1, objective="mse").outputs) == (2, 1)
    with pytest.raises(ValueError, match="n_classes"):
        TabMParams(objective="hl_gauss")
