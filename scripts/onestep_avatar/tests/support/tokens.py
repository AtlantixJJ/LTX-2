"""Share a unique per-test attempt token with plain helper functions."""

from contextvars import ContextVar
from uuid import uuid4

_current = ContextVar("avatar_test_attempt_token", default=None)


def token() -> object:
    value = _current.get()
    if value is None:
        value = uuid4().hex
        _current.set(value)
    return value


def reset_token(value: object) -> object:
    return _current.set(value)


def restore_token(handle: object) -> object:
    _current.reset(handle)
