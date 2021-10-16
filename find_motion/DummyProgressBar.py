class DummyProgressBar(object):
    """
    A pretend progress bar
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self, *args, **kwargs) -> object:
        return self

    def update(self, *args, **kwargs) -> None:
        pass
