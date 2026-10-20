"""Tests for visual DSL framework adapters."""

import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.html_dsl import HtmlDSLAdapter
from ml_switcheroo.frameworks.latex_dsl import LatexDSLAdapter
from ml_switcheroo.frameworks.tikz import TikzAdapter


def test_html_dsl_adapter() -> None:
  """Verify html dsl adapter behavior."""
  adapter: HtmlDSLAdapter = HtmlDSLAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.import_alias == ("html_dsl", "dsl")
  assert "html_dsl" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers

  st = adapter.structural_traits
  assert st.module_base == "html_dsl.Module"
  assert st.forward_method == "forward"
  assert st.init_method_name == "__init__"
  assert st.requires_super_init is True

  assert adapter.plugin_traits.requires_functional_state is False  # default
  adapter.test_config
  adapter.specifications
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []

  with patch("ml_switcheroo.frameworks.html_dsl.load_definitions", return_value={"test": MagicMock()}):
    defs = adapter.definitions
    assert isinstance(defs, dict)

  pass
  assert adapter.convert([1, 2]) == "[1, 2]"
  assert adapter.get_device_syntax("cpu") == ""
  pass
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("load", "path") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert "not supported" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert "not supported" in adapter.get_weight_save_code("s", "p")

  adapter.apply_wiring({})  # just pass
  assert adapter.get_doc_url("html_dsl.Module") is None

  examples: typing.Dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples

  parser = adapter.create_parser("<html>")
  assert parser is not None


def test_latex_dsl_adapter() -> None:
  """Verify latex dsl adapter behavior."""
  adapter: LatexDSLAdapter = LatexDSLAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.import_alias == ("midl", "midl")
  assert "midl" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers

  st = adapter.structural_traits
  assert st.module_base == "midl.Module"
  assert st.requires_super_init is True

  adapter.test_config
  adapter.specifications
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []

  with patch("ml_switcheroo.frameworks.latex_dsl.load_definitions", return_value={"test": MagicMock()}):
    defs = adapter.definitions
    assert isinstance(defs, dict)

  pass
  assert adapter.convert([1, 2]) == "[1, 2]"
  assert adapter.get_device_syntax("cpu") == ""
  pass
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("load", "path") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert "not supported" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert "not supported" in adapter.get_weight_save_code("s", "p")

  adapter.apply_wiring({})
  assert adapter.get_doc_url("latex_dsl.Module") is None

  examples: typing.Dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples

  pass


def test_tikz_adapter() -> None:
  """Verify tikz adapter behavior."""
  adapter: TikzAdapter = TikzAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.import_alias == ("tikz", "tikz")
  assert adapter.import_namespaces == {}
  assert SemanticTier.NEURAL in adapter.supported_tiers

  st = adapter.structural_traits
  assert st.module_base is None

  adapter.test_config
  adapter.specifications
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []

  with patch("ml_switcheroo.frameworks.tikz.load_definitions", return_value={"test": MagicMock()}):
    defs = adapter.definitions
    assert isinstance(defs, dict)

  pass
  assert adapter.convert([1, 2]) == "[1, 2]"
  assert adapter.get_device_syntax("cpu") == ""
  pass
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("load", "path") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert "not supported" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert "not supported" in adapter.get_weight_save_code("s", "p")

  adapter.apply_wiring({})
  assert adapter.get_doc_url("tikz.Module") is None

  examples: typing.Dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples

  pass


def test_latex_parser_create() -> None:
  """Verify latex parser create behavior."""
  adapter: LatexDSLAdapter = LatexDSLAdapter()
  assert adapter.create_parser("code") is not None
  adapter.plugin_traits


def test_tikz_parser_create() -> None:
  """Verify tikz parser create behavior."""
  adapter: TikzAdapter = TikzAdapter()
  assert adapter.plugin_traits is not None


def test_missing() -> None:
  """Verify missing behavior."""
  assert HtmlDSLAdapter().get_device_check_syntax() == "False"
  assert LatexDSLAdapter().get_device_check_syntax() == "True"
  _res = TikzAdapter().get_device_check_syntax()
  assert _res == "True"
