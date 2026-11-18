import pytest

from idun.errors import BackendUnavailableError, CacheError, IdunError, KernelCompileError, NotFittedError


@pytest.mark.parametrize("error", [BackendUnavailableError, KernelCompileError, CacheError, NotFittedError])
def test_every_error_is_an_idun_error(error: type[Exception]) -> None:
    assert issubclass(error, IdunError)
    assert issubclass(IdunError, Exception)


def test_message_is_kept() -> None:
    with pytest.raises(IdunError, match="no Metal device"):
        raise BackendUnavailableError("no Metal device")


def test_not_fitted_is_also_what_scikit_learn_catches() -> None:
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)
