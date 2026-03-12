import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.base import StandardCategory


@patch.dict(
  "sys.modules",
  {
    "mlx": MagicMock(),
    "mlx.core": MagicMock(),
    "mlx.nn": MagicMock(),
    "mlx.nn.losses": MagicMock(),
    "mlx.optimizers": MagicMock(),
  },
)
def test_mlx_adapter_gap():
  adapter = MLXAdapter()

  adapter.search_modules
  adapter.unsafe_submodules
  adapter.plugin_traits

  import mlx.nn
  import mlx.optimizers
  import mlx.nn.losses

  mlx.nn.Layer = type("Layer", (), {})
  mlx.nn.losses.loss = lambda: None
  mlx.optimizers.Opt = type("Opt", (), {})

  try:
    adapter.collect_api(StandardCategory.LAYER)
    adapter.collect_api(StandardCategory.ACTIVATION)
    adapter.collect_api(StandardCategory.LOSS)
    adapter.collect_api(StandardCategory.OPTIMIZER)
  except Exception:
    pass

  try:
    import numpy as np

    adapter.convert(np.array([1, 2, 3]))
  except Exception:
    pass

  adapter.get_serialization_syntax("save", "file", "obj")
  adapter.get_serialization_syntax("load", "file")
  adapter.get_serialization_syntax("unknown", "file")

  adapter.get_tensor_to_numpy_expr("tensor")
  adapter.get_weight_save_code("state", "path")
  adapter.get_weight_load_code("path")
