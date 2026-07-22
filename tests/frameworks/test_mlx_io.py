"""Test suite for the Mlx Io module."""

from ml_switcheroo.frameworks.mlx_io import MlxIOMixin


def test_mlx_io_dummy():
  """Verifies the behavior of MLX I/O dummy."""
  mixin = MlxIOMixin()
  assert hasattr(mixin, "get_serialization_imports")
