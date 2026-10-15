"""Tests for MLIR and StableHLO framework adapters."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.mlir import MlirAdapter
from ml_switcheroo.frameworks.stablehlo import StableHloAdapter


def test_mlir_adapter_properties() -> None:
  """Verify mlir adapter properties behavior."""
  adapter: MlirAdapter = MlirAdapter()
  assert adapter.import_alias == ("mlir", "sw")
  assert isinstance(adapter.import_namespaces, dict)
  assert isinstance(adapter.supported_tiers, list)
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  assert isinstance(adapter.test_config, dict)
  assert isinstance(adapter.harness_imports, list)
  assert isinstance(adapter.get_harness_init_code(), str)
  assert isinstance(adapter.get_to_numpy_code(), str)
  assert isinstance(adapter.declared_magic_args, list)
  assert isinstance(adapter.rng_seed_methods, list)

  with patch("ml_switcheroo.frameworks.mlir.load_definitions", return_value={"test": MagicMock()}):
    assert "test" in adapter.definitions

  assert isinstance(adapter.specifications, dict)
  assert adapter.convert([1, 2]) == "[1, 2]"
  assert isinstance(adapter.get_device_syntax("cpu"), str)
  assert isinstance(adapter.get_device_check_syntax(), str)
  assert isinstance(adapter.get_rng_split_syntax("rng", "key"), str)
  assert isinstance(adapter.get_serialization_imports(), list)
  assert isinstance(adapter.get_serialization_syntax("load", "path"), str)
  assert isinstance(adapter.get_weight_conversion_imports(), list)
  assert isinstance(adapter.get_weight_load_code("path"), str)
  assert isinstance(adapter.get_tensor_to_numpy_expr("t"), str)
  assert isinstance(adapter.get_weight_save_code("s", "path"), str)

  wiring: dict[str, object] = {}
  adapter.apply_wiring(wiring)
  assert wiring == {}
  assert adapter.get_doc_url("mlir.Module") is None


def test_stablehlo_adapter_properties() -> None:
  """Verify stablehlo adapter properties behavior."""
  adapter: StableHloAdapter = StableHloAdapter()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert isinstance(adapter.import_namespaces, dict)
  assert isinstance(adapter.supported_tiers, list)
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  assert isinstance(adapter.test_config, dict)
  assert isinstance(adapter.harness_imports, list)
  assert isinstance(adapter.get_harness_init_code(), str)
  assert isinstance(adapter.get_to_numpy_code(), str)
  assert isinstance(adapter.declared_magic_args, list)
  assert isinstance(adapter.rng_seed_methods, list)

  with patch("ml_switcheroo.frameworks.stablehlo.load_definitions", return_value={"test": MagicMock()}):
    assert "test" in adapter.definitions

  assert isinstance(adapter.specifications, dict)
  assert adapter.convert([1, 2]) == "[1, 2]"
  assert isinstance(adapter.get_device_syntax("cpu"), str)
  assert isinstance(adapter.get_device_check_syntax(), str)
  assert isinstance(adapter.get_rng_split_syntax("rng", "key"), str)
  assert isinstance(adapter.get_serialization_imports(), list)
  assert isinstance(adapter.get_serialization_syntax("load", "path"), str)
  assert isinstance(adapter.get_weight_conversion_imports(), list)
  assert isinstance(adapter.get_weight_load_code("path"), str)
  assert isinstance(adapter.get_tensor_to_numpy_expr("t"), str)
  assert isinstance(adapter.get_weight_save_code("s", "path"), str)

  wiring: dict[str, object] = {}
  adapter.apply_wiring(wiring)
  assert wiring == {}
  assert adapter.get_doc_url("stablehlo.Module") == "https://github.com/openxla/stablehlo/blob/main/docs/spec.md#Module"


def test_mlir_stablehlo_missing_methods() -> None:
  """Verify mlir stablehlo missing methods behavior."""
  adapter: MlirAdapter = MlirAdapter()
  assert isinstance(adapter.get_serialization_syntax("save", "path", "obj"), str)
  assert isinstance(adapter.get_example_code(), str)
  assert isinstance(adapter.get_tiered_examples(), dict)

  adapter2: StableHloAdapter = StableHloAdapter()
  assert isinstance(adapter2.get_serialization_syntax("save", "path", "obj"), str)
  assert adapter2.get_doc_url("test") is None
  assert isinstance(adapter2.get_tiered_examples(), dict)
