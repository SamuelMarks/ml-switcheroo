"""Test module."""

from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_html_dsl_missing_methods():
  """Test function."""
  adapter = HtmlDSLAdapter()
  assert isinstance(adapter.plugin_traits, PluginTraits)
  assert adapter.convert("test") == "test"
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("load", "file") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in HTML mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in HTML mode"
