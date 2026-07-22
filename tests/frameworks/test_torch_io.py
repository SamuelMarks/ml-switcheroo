"""Test suite for the Torch Io module."""

from ml_switcheroo.frameworks.torch_io import TorchIOMixin


def test_torch_io_dummy():
  """Verifies the behavior of PyTorch I/O dummy."""
  mixin = TorchIOMixin()
  assert hasattr(mixin, "get_serialization_imports")
