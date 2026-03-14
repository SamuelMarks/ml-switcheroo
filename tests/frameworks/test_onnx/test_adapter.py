"""Test ONNX framework adapter."""

from ml_switcheroo.frameworks.onnx.adapter import OnnxFramework
from ml_switcheroo.frameworks.base import get_adapter
import ml_switcheroo.frameworks.onnx


def test_onnx_adapter_discovery():
  adapter = get_adapter("onnx")
  assert adapter is not None
  assert isinstance(adapter, OnnxFramework)
  assert adapter.display_name == "ONNX"


from ml_switcheroo.frameworks.onnx.adapter import OnnxFramework
from ml_switcheroo.enums import SemanticTier


def test_onnx_adapter_properties():
  adapter = OnnxFramework()
  assert adapter.display_name == "ONNX"
  assert adapter.ui_priority == 5
  assert adapter.inherits_from is None
  assert isinstance(adapter.search_modules, list)
  assert isinstance(adapter.unsafe_submodules, set)
  assert adapter.import_alias == ("onnx", "onnx")

  namespaces = adapter.import_namespaces
  assert "onnx" in namespaces
  assert "onnx.helper" in namespaces
  assert namespaces["onnx"].tier == SemanticTier.EXTRAS
  assert namespaces["onnx.helper"].tier == SemanticTier.ARRAY_API

  heuristics = adapter.discovery_heuristics
  assert "extras" in heuristics

  test_config = adapter.test_config
  assert "import" in test_config
  assert "convert_input" in test_config
  assert "to_numpy" in test_config
  assert "jit_template" in test_config

  assert isinstance(adapter.harness_imports, list)
  assert isinstance(adapter.get_harness_init_code(), str)
  assert isinstance(adapter.declared_magic_args, list)
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None
  assert isinstance(adapter.rng_seed_methods, list)
  assert isinstance(adapter.definitions, dict)
  assert isinstance(adapter.specifications, dict)

  assert adapter.collect_api(None) == []
  adapter.apply_wiring({})
  assert adapter.convert("test") == "test"

  assert adapter.get_device_syntax("cuda") == "None"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  assert isinstance(adapter.get_serialization_imports(), list)
  assert adapter.get_serialization_syntax("save", "file.onnx", "model") == "onnx.save(model, file.onnx)"
  assert adapter.get_serialization_syntax("load", "file.onnx") == "onnx.load(file.onnx)"
  assert adapter.get_serialization_syntax("other", "file.onnx") == ""

  assert isinstance(adapter.get_weight_conversion_imports(), list)
  assert isinstance(adapter.get_weight_load_code("path"), str)
  assert isinstance(adapter.get_tensor_to_numpy_expr("tensor"), str)
  assert isinstance(adapter.get_weight_save_code("state", "path"), str)
  assert isinstance(adapter.get_doc_url("test"), str)
  assert isinstance(adapter.get_tiered_examples(), dict)
  assert isinstance(adapter.get_to_numpy_code(), str)
