"""Test suite for the Numpy module."""


def setup_adam(model, lr: float = 0.001):
  """Helper to setup adam."""

  class NumpyAdam:
    """Test suite for the Numpy Adam component."""

    def __init__(self, lr):
      """Initializes the NumpyAdam instance."""
      self.lr = lr

  return NumpyAdam(lr)
