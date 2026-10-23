"""Test suite for the Doc Context module."""

from unittest.mock import MagicMock, patch
from pydantic import JsonValue

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_context import DocContextBuilder


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Execute mock semantics helper logic.

  Returns:
      Configured `mock_semantics` fixture value (MagicMock).
  """
  return MagicMock(spec=SemanticsManager)


@pytest.fixture
def builder(mock_semantics: MagicMock) -> DocContextBuilder:
  """Execute builder helper logic.

  Args:
      mock_semantics: Mocked SemanticsManager instance.

  Returns:
      Configured `builder` fixture value (DocContextBuilder).
  """
  return DocContextBuilder(mock_semantics)


def test_argument_formatting_string(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting string.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  std_args = ["x", "y"]
  formatted: list[str] = builder._format_args(std_args)
  assert formatted == ["x", "y"]


def test_argument_formatting_tuple(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting tuple.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  std_args = [("x", "Tensor"), "dim"]
  formatted: list[str] = builder._format_args(std_args)
  assert formatted == ["x: Tensor", "dim"]


def test_argument_formatting_dict(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting dictionary.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  std_args = [
    {"name": "dim", "type": "int", "default": "-1"},
    {"name": "__doc__", "type": "str", "default": "docstring"},
    {"name": "clean_backticks", "default": "'```(None)```'"},
  ]
  formatted: list[str] = builder._format_args(std_args)
  assert formatted == ["dim: int = -1", "clean_backticks = '(None)'"]


def test_argument_formatting_edge_branches(builder: DocContextBuilder) -> None:
  """Verifies branches with Any type hint, default without type, and tuple length 1.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  std_args = [
    {"name": "any_arg", "type": "Any", "default": "None"},
    {"name": "any_no_def", "type": "Any"},
    {"name": "val_only", "default": 42},
    ("single_elem",),
    (),
  ]
  formatted: list[str] = builder._format_args(std_args)
  assert formatted == ["any_arg = None", "any_no_def", "val_only = 42", "single_elem", "unknown"]


def test_missing_property_defaults(builder: DocContextBuilder) -> None:
  """Verifies the behavior of missing property defaults.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  context = builder.build("EmptyOp", {})
  assert context["name"] == "EmptyOp"
  assert context["description"] == "No description available."
  assert context["args"] == []
  assert context["variants"] == []


def test_impl_type_classification_plugin(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification plugin.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  var = {"requires_plugin": "my_hook"}
  assert builder._determine_impl_type(var) == "Plugin (my_hook)"


def test_impl_type_classification_macro(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification macro.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  var = {"macro_template": "{x}*2"}
  assert builder._determine_impl_type(var) == "Macro '{x}*2'"


def test_impl_type_classification_infix(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification infix.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  var = {"transformation_type": "infix", "operator": "+"}
  assert builder._determine_impl_type(var) == "Infix (+)"


def test_impl_type_classification_direct(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification direct.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  var = {"api": "torch.abs"}
  assert builder._determine_impl_type(var) == "Direct Mapping"


def test_impl_type_classification_inline_lambda_and_fallback(builder: DocContextBuilder) -> None:
  """Verifies the behavior of inline_lambda and fallback custom classification.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  var_lambda: dict[str, JsonValue] = {"transformation_type": "inline_lambda"}
  assert builder._determine_impl_type(var_lambda) == "Inline Lambda"

  var_custom: dict[str, JsonValue] = {}
  assert builder._determine_impl_type(var_custom) == "Custom / Partial"


def test_resolve_variants_skips_none_and_handles_macro(builder: DocContextBuilder) -> None:
  """Verifies that null variants are skipped and macros do not get doc URLs.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  op_def: dict[str, JsonValue] = {
    "variants": {
      "torch": None,
      "jax": {"macro_template": "{x} * 2", "api": "jax.numpy.multiply"},
    }
  }
  mock_jax: MagicMock = MagicMock()
  mock_jax.display_name = "JAX"
  with patch("ml_switcheroo.utils.doc_context.get_adapter", return_value=mock_jax):
    context = builder.build("Op", op_def)
  variants = context["variants"]
  assert isinstance(variants, list)
  assert len(variants) == 1
  v0 = variants[0]
  assert isinstance(v0, dict)
  assert v0["key"] == "jax"
  assert v0["doc_url"] is None


def test_full_build_flow_with_adapter_logic(builder: DocContextBuilder) -> None:
  """Verifies the behavior of full build flow with adapter logic.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  op_def: dict[str, JsonValue] = {
    "description": "Calculate abs.",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"requires_plugin": "magic"}, "unknown_fw": {"api": "foo"}},
  }
  mock_torch: MagicMock = MagicMock()
  mock_torch.display_name = "PyTorch"
  mock_torch.get_doc_url.return_value = "http://torch/abs"
  mock_jax: MagicMock = MagicMock()
  mock_jax.display_name = "JAX"

  def get_adapter_side_effect(name: str) -> MagicMock | None:
    """Gets adapter side effect.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `MagicMock | None` from get adapter side effect.
    """
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  with patch("ml_switcheroo.utils.doc_context.get_framework_priority_order", return_value=["torch", "jax"]):
    with patch("ml_switcheroo.utils.doc_context.get_adapter", side_effect=get_adapter_side_effect):
      context = builder.build("Abs", op_def)
  assert context["name"] == "Abs"
  assert context["description"] == "Calculate abs."
  assert context["args"] == ["x"]
  variants = context["variants"]
  assert isinstance(variants, list)
  assert len(variants) == 3
  v0 = variants[0]
  assert isinstance(v0, dict)
  assert v0["key"] == "torch"
  assert v0["framework"] == "PyTorch"
  assert v0["api"] == "torch.abs"
  assert v0["doc_url"] == "http://torch/abs"
  assert v0["implementation_type"] == "Direct Mapping"
  v1 = variants[1]
  assert isinstance(v1, dict)
  assert v1["key"] == "jax"
  assert v1["implementation_type"] == "Plugin (magic)"
  assert v1["doc_url"] is None
  v2 = variants[2]
  assert isinstance(v2, dict)
  assert v2["key"] == "unknown_fw"
  assert v2["framework"] == "unknown_fw"


def test_resolve_variants_adapter_without_display_name(builder: DocContextBuilder) -> None:
  """Verifies fallback when adapter lacks display_name attribute.

  Args:
      builder: Runtime or IR builder helper instance.
  """
  op_def: dict[str, JsonValue] = {"variants": {"custom": {"api": "custom.api"}}}
  mock_adapter = MagicMock(spec=["get_doc_url"])  # No display_name attribute
  mock_adapter.get_doc_url.return_value = "http://doc"
  with patch("ml_switcheroo.utils.doc_context.get_adapter", return_value=mock_adapter):
    context = builder.build("Op", op_def)
  variants = context["variants"]
  assert isinstance(variants, list)
  v0 = variants[0]
  assert isinstance(v0, dict)
  assert v0["framework"] == "custom"
