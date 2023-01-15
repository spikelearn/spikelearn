from collections import deque

class Layer:
    """Base class for a Layer

    Provides the basic interface that any layer instance should have.
    """

    def __call__(self, *x):
        raise NotImplementedError()

    def reset(self):
        pass

    @property
    def out(self):
        raise NotImplementedError()

