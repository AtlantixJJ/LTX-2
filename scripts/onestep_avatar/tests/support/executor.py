"""Synchronous pool substitute for publication/cache tests, not spawn checks."""


class InlineExecutor:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> object:
        return self

    def __exit__(self, *args: object) -> object:
        return False

    def map(self, function: object, *iterables: object, **_kwargs: object) -> object:
        return map(function, *iterables)
