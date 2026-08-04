"""Test module."""

from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_latex_dsl_adapter_missing_methods():
  """Test function."""
  adapter = LatexDSLAdapter()
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in LaTeX mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in LaTeX mode"
  assert adapter.convert("data") == "data"

  assert isinstance(adapter.plugin_traits, PluginTraits)
