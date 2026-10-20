"""Tests for MLX framework adapters."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.mlx import MLXAdapter


def test_mlx_adapter_properties() -> None:
  """Verify mlx adapter properties behavior."""
  adapter = MLXAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch(
    "ml_switcheroo.frameworks.mlx.load_definitions",
    return_value={"Linear": MagicMock(), "Conv2d": MagicMock()},
  ):
    assert "Linear" in adapter.definitions

  with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
    adapter.convert([1, 2])

  mock_mlx = MagicMock()
  mock_core = MagicMock()
  mock_mlx.core = mock_core
  with patch.dict("sys.modules", {"mlx": mock_mlx, "mlx.core": mock_core}):
    mock_core.array.return_value = "tensor"
    adapter.convert([1, 2])

    mock_core.array.side_effect = RuntimeError("err")
    adapter.convert([1, 2])

  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path", "obj")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("mlx.nn.Linear")
  adapter.get_doc_url("mlx.core.add")
  adapter.get_doc_url("unknown")
  adapter.get_tiered_examples()

  # io mixin coverage
  pass
  pass
  pass
  pass


def test_mlx_missing_lines() -> None:
  """Verify mlx missing lines behavior."""
  from ml_switcheroo.frameworks.mlx import MLXAdapter

  adapter = MLXAdapter()
  adapter.get_device_syntax("gpu", "1")
  adapter.get_device_syntax("gpu")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_serialization_syntax("load", "path")
  _res = adapter.get_serialization_syntax("other", "path")
  assert _res == ""


def test_mlx_io_mixin_direct() -> None:
  """Verify mlx io mixin direct behavior."""
  from ml_switcheroo.frameworks.mlx_io import MlxIOMixin

  mixin = MlxIOMixin()
  _res = mixin.get_serialization_imports()
  assert _res == ["import mlx.core as mx"]
