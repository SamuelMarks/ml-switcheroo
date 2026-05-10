from unittest import mock


def test_mlx_imports_success():
  mock_mlx = mock.MagicMock()
  with mock.patch.dict(
    "sys.modules",
    {"mlx": mock_mlx, "mlx.core": mock_mlx, "mlx.nn": mock_mlx, "mlx.optimizers": mock_mlx, "mlx.utils": mock_mlx},
  ):
    import ml_switcheroo.frameworks.mlx as fmlx
    import importlib

    importlib.reload(fmlx)
