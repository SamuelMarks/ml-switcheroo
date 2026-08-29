"""Test suite for the Doc Context module."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_context import DocContextBuilder


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Docstring."""
  return MagicMock(spec=SemanticsManager)


@pytest.fixture
def builder(mock_semantics: MagicMock) -> DocContextBuilder:
  """Docstring."""
  return DocContextBuilder(mock_semantics)


def test_argument_formatting_string(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting string."""
  std_args: List[Any] = ["x", "y"]
  formatted: List[str] = builder._format_args(std_args)
  assert formatted == ["x", "y"]


def test_argument_formatting_tuple(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting tuple."""
  std_args: List[Any] = [("x", "Tensor"), "dim"]
  formatted: List[str] = builder._format_args(std_args)
  assert formatted == ["x: Tensor", "dim"]


def test_argument_formatting_dict(builder: DocContextBuilder) -> None:
  """Verifies the behavior of argument formatting dictionary."""
  std_args: List[Any] = [{"name": "dim", "type": "int", "default": "-1"}]
  formatted: List[str] = builder._format_args(std_args)
  assert formatted == ["dim: int = -1"]


def test_missing_property_defaults(builder: DocContextBuilder) -> None:
  """Verifies the behavior of missing property defaults."""
  context: Dict[str, Any] = builder.build("EmptyOp", {})
  assert context["name"] == "EmptyOp"
  assert context["description"] == "No description available."
  assert context["args"] == []
  assert context["variants"] == []


def test_impl_type_classification_plugin(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification plugin."""
  var: Dict[str, Any] = {"requires_plugin": "my_hook"}
  assert builder._determine_impl_type(var) == "Plugin (my_hook)"


def test_impl_type_classification_macro(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification macro."""
  var: Dict[str, Any] = {"macro_template": "{x}*2"}
  assert builder._determine_impl_type(var) == "Macro '{x}*2'"


def test_impl_type_classification_infix(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification infix."""
  var: Dict[str, Any] = {"transformation_type": "infix", "operator": "+"}
  assert builder._determine_impl_type(var) == "Infix (+)"


def test_impl_type_classification_direct(builder: DocContextBuilder) -> None:
  """Verifies the behavior of impl type classification direct."""
  var: Dict[str, Any] = {"api": "torch.abs"}
  assert builder._determine_impl_type(var) == "Direct Mapping"


def test_full_build_flow_with_adapter_logic(builder: DocContextBuilder) -> None:
  """Verifies the behavior of full build flow with adapter logic."""
  op_def: Dict[str, Any] = {
    "description": "Calculate abs.",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"requires_plugin": "magic"}, "unknown_fw": {"api": "foo"}},
  }
  mock_torch: MagicMock = MagicMock()
  mock_torch.display_name = "PyTorch"
  mock_torch.get_doc_url.return_value = "http://torch/abs"
  mock_jax: MagicMock = MagicMock()
  mock_jax.display_name = "JAX"

  def get_adapter_side_effect(name: str) -> Any:
    """Gets adapter side effect."""
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  with patch("ml_switcheroo.utils.doc_context.get_framework_priority_order", return_value=["torch", "jax"]):
    with patch("ml_switcheroo.utils.doc_context.get_adapter", side_effect=get_adapter_side_effect):
      context: Dict[str, Any] = builder.build("Abs", op_def)
  assert context["name"] == "Abs"
  assert context["description"] == "Calculate abs."
  assert context["args"] == ["x"]
  assert len(context["variants"]) == 3
  v0: Dict[str, Any] = context["variants"][0]
  assert v0["key"] == "torch"
  assert v0["framework"] == "PyTorch"
  assert v0["api"] == "torch.abs"
  assert v0["doc_url"] == "http://torch/abs"
  assert v0["implementation_type"] == "Direct Mapping"
  v1: Dict[str, Any] = context["variants"][1]
  assert v1["key"] == "jax"
  assert v1["implementation_type"] == "Plugin (magic)"
  assert v1["doc_url"] is None
  v2: Dict[str, Any] = context["variants"][2]
  assert v2["key"] == "unknown_fw"
  assert v2["framework"] == "unknown_fw"
