"""Unit tests for `HtmlDSLAdapter` (`ml_switcheroo.frameworks.html_dsl`)."""

from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import ImportConfig, StandardMap
from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo.semantics.schema import PluginTraits


def test_html_dsl_adapter_properties_traits_and_definitions() -> None:
  """Verify `HtmlDSLAdapter` metadata, parser creation, traits, serialization stubs, and definitions."""
  adapter = HtmlDSLAdapter()
  assert adapter.display_name == "HTML Grid DSL"
  assert adapter.ui_priority == 980
  assert adapter.inherits_from is None
  assert adapter.import_alias == ("html_dsl", "dsl")
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.test_config == {}
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []

  mod = adapter.create_parser("<h3>Model: M</h3>").parse()
  assert "class M" in mod.code

  assert adapter.structural_traits.module_base == "html_dsl.Module"
  assert adapter.structural_traits.forward_method == "forward"
  assert isinstance(adapter.plugin_traits, PluginTraits)
  ns_html = adapter.import_namespaces["html_dsl"]
  assert isinstance(ns_html, ImportConfig)
  assert ns_html.tier == SemanticTier.NEURAL

  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.convert(123) == "123"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file.pt") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in HTML mode"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in HTML mode"

  defs = adapter.definitions
  assert "Module" in defs
  assert defs["Conv2d"].api == "html_dsl.Conv2d"
  assert adapter.specifications == {}
  assert adapter.get_doc_url("html_dsl.Module") is None
  assert "conv: Conv2d" in adapter.get_tiered_examples()["tier2_neural"]
  adapter.apply_wiring({})

  custom_defs = {
    "Module": StandardMap(api="custom.Module"),
    "Conv2d": StandardMap(api="custom.Conv2d"),
  }
  with patch("ml_switcheroo.frameworks.html_dsl.load_definitions", return_value=custom_defs):
    pre_defs = HtmlDSLAdapter().definitions
    assert pre_defs["Module"].api == "custom.Module"
    assert pre_defs["Conv2d"].api == "custom.Conv2d"
