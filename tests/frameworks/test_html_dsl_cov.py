import pytest
from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo.enums import SemanticTier


def test_html_dsl_adapter_properties():
  adapter = HtmlDSLAdapter()

  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("html_dsl", "dsl")
  assert "html_dsl" in adapter.import_namespaces
  assert adapter.discovery_heuristics == {}
  assert len(adapter.supported_tiers) > 0
  assert adapter.test_config == {}
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []
  assert len(adapter.definitions) > 0
  assert adapter.specifications == {}

  assert adapter.collect_api(SemanticTier.NEURAL) == []
  assert adapter.convert(None) == "None"

  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == ""

  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("save", "file", "obj") == ""
  assert adapter.get_weight_conversion_imports() == []

  assert adapter.get_weight_load_code("path") == "# Weights not supported in HTML mode"
  assert adapter.get_tensor_to_numpy_expr("tensor") == "tensor"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in HTML mode"

  assert adapter.get_doc_url("api") is None


def test_html_dsl_wiring():
  adapter = HtmlDSLAdapter()
  snap = {}
  adapter.apply_wiring(snap)
  assert snap == {}


def test_html_dsl_examples():
  from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter

  adapter = HtmlDSLAdapter()
  assert len(adapter.get_tiered_examples()) > 0


def test_html_dsl_other_methods():
  adapter = HtmlDSLAdapter()
  assert adapter.create_parser("").__class__.__name__ == "HtmlParser"
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None
  assert adapter.specifications == {}
  assert adapter.collect_api(SemanticTier.NEURAL) == []
  assert adapter.convert("test") == "test"
  assert adapter.get_device_syntax("cpu") == ""
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("op", "f") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("p") == "# Weights not supported in HTML mode"
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("s", "p") == "# Weights not supported in HTML mode"
  assert adapter.apply_wiring({}) is None


def test_html_dsl_doc_url():
  adapter = HtmlDSLAdapter()
  assert adapter.get_doc_url("api") is None
