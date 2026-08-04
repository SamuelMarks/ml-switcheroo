"""Test module."""

from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_rdna_missing_methods():
  """Test function."""
  adapter = RdnaAdapter()

  assert adapter.get_device_syntax("cpu") == "; Target Device: cpu"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == ""
  assert adapter.get_serialization_syntax("load", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "; Weights loading not supported in RDNA adapter"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "; Weights saving not supported in RDNA adapter"

  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.convert("data") == "data"
  assert adapter.ui_priority == 151

  defs = adapter.definitions
  assert isinstance(defs, dict)

  ex = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural_simple" in ex
  assert "tier3_extras" in ex
  pass

  adapter.apply_wiring({})
  assert "gpuopen.com" in adapter.get_doc_url("api")
