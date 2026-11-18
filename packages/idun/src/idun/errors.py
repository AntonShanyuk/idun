"""Exceptions idun raises on purpose.

Idun fails loudly: when a requested capability is missing it raises one of these, and it never substitutes a different
backend, precision or code path behind the caller's back (docs/development/coding-guidelines.md).
"""

__all__ = ["BackendUnavailableError", "CacheError", "IdunError", "KernelCompileError", "MultiTargetQualityWarning",
           "NotFittedError"]  # fmt: skip


class IdunError(Exception):
    """Base class of every error idun raises on purpose."""


class BackendUnavailableError(IdunError):
    """The requested backend cannot run in this process: wrong platform, missing native library or no device."""


class KernelCompileError(IdunError):
    """A device kernel failed to compile. The message is the compiler log."""


class CacheError(IdunError):
    """The packed data cache is missing, stale, corrupt or of another format version. It is never rebuilt silently."""


class NotFittedError(IdunError, ValueError, AttributeError):
    """An estimator was asked to predict or save before ``fit``.

    It is a ``ValueError`` and an ``AttributeError``, as scikit-learn's own class is (which idun does not import).
    """


class MultiTargetQualityWarning(UserWarning):
    """A model was asked to fit several targets at once (experimental, not covered by the documentation)."""
