import pytest

from idun.gbm import Params


def test_defaults_are_valid_and_round_trip() -> None:
    params = Params()
    assert Params.from_dict(params.to_dict()) == params
    assert not params.bagging


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_trees", 0),
        ("learning_rate", 0.0),
        ("max_depth", 9),
        ("max_depth", -1),
        ("growth", "bestfirst"),
        ("max_leaves", 1),
        ("max_leaves", 65),
        ("min_data_in_leaf", 0),
        ("lambda_l2", -1.0),
        ("feature_fraction", 0.0),
        ("bagging_fraction", 1.5),
        ("bagging_freq", -1),
        ("grad_quant_bins", 3),
        ("categorical", "onehot"),
        ("cat_smooth", -1.0),
        ("cat_l2", -1.0),
        ("max_cat_threshold", 0),
        ("min_data_per_group", 0),
        ("seed", 2**32),
    ],
)
def test_out_of_range_values_name_the_field(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"invalid Params.{field}"):
        Params(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"), [("num_trees", 1.5), ("lambda_l2", True), ("categorical", 1), ("seed", "0")]
)
def test_wrong_types_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"Params.{field} has the wrong type"):
        Params(**{field: value})  # type: ignore[arg-type]


def test_integers_are_accepted_for_floats() -> None:
    assert Params(lambda_l2=5, learning_rate=1).lambda_l2 == 5


def test_bagging_needs_a_fraction_and_a_frequency() -> None:
    assert Params(bagging_fraction=0.8, bagging_freq=1).bagging
    assert not Params(bagging_fraction=0.8, bagging_freq=0).bagging
    assert not Params(bagging_fraction=1.0, bagging_freq=1).bagging


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Params fields"):
        Params.from_dict({"num_leaves": 63})


@pytest.mark.parametrize(
    ("depth", "leaves"), [(-1, 2), (-1, 63), (-1, 256), (1, 2), (6, 64), (8, 256), (30, 255), (100, 256)]
)
def test_leafwise_takes_any_depth_or_none_and_up_to_256_leaves(depth: int, leaves: int) -> None:
    params = Params(growth="leafwise", max_depth=depth, max_leaves=leaves)
    assert params.leafwise
    assert Params.from_dict(params.to_dict()) == params


@pytest.mark.parametrize(("field", "depth", "leaves"), [("max_depth", 0, 4), ("max_depth", -2, 4),
                                                        ("max_leaves", -1, 257), ("max_leaves", 3, 9),
                                                        ("max_leaves", -1, 1)])  # fmt: skip
def test_leafwise_limits_name_the_field(field: str, depth: int, leaves: int) -> None:
    with pytest.raises(ValueError, match=f"invalid Params.{field}"):
        Params(growth="leafwise", max_depth=depth, max_leaves=leaves)


def test_growth_defaults_to_depthwise_and_old_dicts_load() -> None:
    assert Params().growth == "depthwise"
    assert not Params().leafwise
    old = {k: v for k, v in Params().to_dict().items() if k != "growth"}  # a model file without growth
    assert Params.from_dict(old) == Params()
