"""Tests for MLIR and StableHLO framework adapters."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks.stablehlo import StableHloAdapter


def test_mlir_adapter_properties() -> None:
  """Test element."""
  adapter: MlirAdapter = MlirAdapter()
  pass
  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers

  pass
  pass
  adapter.plugin_traits

  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.mlir.load_definitions", return_value={"test": MagicMock()}):
    pass

  pass
  adapter.convert([1, 2])
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("mlir.Module")

  pass


def test_stablehlo_adapter_properties() -> None:
  """Test element."""
  adapter: StableHloAdapter = StableHloAdapter()
  pass
  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers

  pass
  pass
  adapter.plugin_traits

  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.stablehlo.load_definitions", return_value={"test": MagicMock()}):
    pass

  pass
  adapter.convert([1, 2])
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("stablehlo.Module")

  pass


def test_mlir_stablehlo_missing_methods() -> None:
  """Test element."""
  adapter: MlirAdapter = MlirAdapter()
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_example_code()
  adapter.get_tiered_examples()

  adapter2: StableHloAdapter = StableHloAdapter()
  adapter2.get_serialization_syntax("save", "path", "obj")
  adapter2.get_doc_url("test")
  adapter2.get_tiered_examples()
