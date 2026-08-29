"""Test suite for the Frameworks Mlx Gap module."""

from unittest import mock


def test_mlx_imports_success() -> None:
  """Verifies the behavior of MLX imports successfully."""
  mock_mlx: mock.MagicMock = mock.MagicMock()
  with mock.patch.dict(
    "sys.modules",
    {"mlx": mock_mlx, "mlx.core": mock_mlx, "mlx.nn": mock_mlx, "mlx.optimizers": mock_mlx, "mlx.utils": mock_mlx},
  ):
    import importlib

    import ml_switcheroo.frameworks.mlx as fmlx

    importlib.reload(fmlx)
