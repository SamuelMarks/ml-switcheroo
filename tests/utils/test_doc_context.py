"""Test suite for the Doc Context module."""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.utils.doc_context import DocContextBuilder
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  return MagicMock(spec=SemanticsManager)


@pytest.fixture
def builder(mock_semantics):
  """Provides a mock builder for testing."""
  return DocContextBuilder(mock_semantics)


def test_argument_formatting_string(builder):
  """Verifies the behavior of argument formatting string."""
  std_args = ["x", "y"]
  formatted = builder._format_args(std_args)
  assert formatted == ["x", "y"]


def test_argument_formatting_tuple(builder):
  """Verifies the behavior of argument formatting tuple."""
  std_args = [("x", "Tensor"), "dim"]
  formatted = builder._format_args(std_args)
  assert formatted == ["x: Tensor", "dim"]


def test_argument_formatting_dict(builder):
  """Verifies the behavior of argument formatting dictionary."""
  std_args = [{"name": "dim", "type": "int", "default": "-1"}]
  formatted = builder._format_args(std_args)
  assert formatted == ["dim: int = -1"]


def test_missing_property_defaults(builder):
  """Verifies the behavior of missing property defaults."""
  context = builder.build("EmptyOp", {})
  assert context["name"] == "EmptyOp"
  assert context["description"] == "No description available."
  assert context["args"] == []
  assert context["variants"] == []


def test_impl_type_classification_plugin(builder):
  """Verifies the behavior of impl type classification plugin."""
  var = {"requires_plugin": "my_hook"}
  assert builder._determine_impl_type(var) == "Plugin (my_hook)"


def test_impl_type_classification_macro(builder):
  """Verifies the behavior of impl type classification macro."""
  var = {"macro_template": "{x}*2"}
  assert builder._determine_impl_type(var) == "Macro '{x}*2'"


def test_impl_type_classification_infix(builder):
  """Verifies the behavior of impl type classification infix."""
  var = {"transformation_type": "infix", "operator": "+"}
  assert builder._determine_impl_type(var) == "Infix (+)"


def test_impl_type_classification_direct(builder):
  """Verifies the behavior of impl type classification direct."""
  var = {"api": "torch.abs"}
  assert builder._determine_impl_type(var) == "Direct Mapping"


def test_full_build_flow_with_adapter_logic(builder):
  """Verifies the behavior of full build flow with adapter logic."""
  op_def = {
    "description": "Calculate abs.",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"requires_plugin": "magic"}, "unknown_fw": {"api": "foo"}},
  }
  mock_torch = MagicMock()
  mock_torch.display_name = "PyTorch"
  mock_torch.get_doc_url.return_value = "http://torch/abs"
  mock_jax = MagicMock()
  mock_jax.display_name = "JAX"

  def get_adapter_side_effect(name):
    """Gets adapter side effect."""
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
  assert len(context["variants"]) == 3
  v0 = context["variants"][0]
  assert v0["key"] == "torch"
  assert v0["framework"] == "PyTorch"
  assert v0["api"] == "torch.abs"
  assert v0["doc_url"] == "http://torch/abs"
  assert v0["implementation_type"] == "Direct Mapping"
  v1 = context["variants"][1]
  assert v1["key"] == "jax"
  assert v1["implementation_type"] == "Plugin (magic)"
  assert v1["doc_url"] is None
  v2 = context["variants"][2]
  assert v2["key"] == "unknown_fw"
  assert v2["framework"] == "unknown_fw"
