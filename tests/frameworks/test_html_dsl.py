"""Tests for HTML DSL framework adapter."""

from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_html_dsl_adapter_basics():
  """Test standard properties and basic behavior."""
  adapter = HtmlDSLAdapter()
  assert adapter.display_name == "HTML Grid DSL"
  assert adapter.import_alias == ("html_dsl", "dsl")
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.test_config == {}
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []


def test_html_dsl_parser():
  """Test parser creation."""
  adapter = HtmlDSLAdapter()
  parser = adapter.create_parser("<div></div>")
  # Assuming HtmlParser just takes the code, checking if it doesn't fail
  assert parser is not None


def test_html_dsl_traits_and_namespaces():
  """Test structural and plugin traits and namespaces."""
  adapter = HtmlDSLAdapter()
  assert adapter.structural_traits.module_base == "html_dsl.Module"
  assert adapter.structural_traits.forward_method == "forward"
  assert adapter.plugin_traits is not None

  namespaces = adapter.import_namespaces
  assert "html_dsl" in namespaces
  assert namespaces["html_dsl"].tier == SemanticTier.NEURAL


def test_html_dsl_device_and_serialization():
  """Test device, conversion and serialization methods."""
  adapter = HtmlDSLAdapter()

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


def test_html_dsl_definitions_and_docs():
  """Test definitions, specifications and examples."""
  adapter = HtmlDSLAdapter()

  defs = adapter.definitions
  assert "Module" in defs
  assert "Conv2d" in defs
  assert defs["Conv2d"].api == "html_dsl.Conv2d"

  assert adapter.specifications == {}
  assert adapter.get_doc_url("html_dsl.Module") is None

  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "conv: Conv2d" in examples["tier2_neural"]

  # Test apply_wiring doesn't crash
  adapter.apply_wiring({})
