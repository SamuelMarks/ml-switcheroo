"""
Tests for DocContextBuilder.

Verifies:
1.  Construction of view model from semantic definitions.
2.  Argument formatting including types and defaults.
3.  Variant classification (Plugin/Macro/Direct).
4.  URL resolution via adapter mocking.
"""

import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.utils.doc_context import DocContextBuilder
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Returns a mock manager."""
  return MagicMock(spec=SemanticsManager)


@pytest.fixture
def builder(mock_semantics):
  """Returns the builder instance."""
  return DocContextBuilder(mock_semantics)


def test_argument_formatting_string(builder):
  """
  Scenario: ["x", "y"]
  """
  std_args = ["x", "y"]
  formatted = builder._format_args(std_args)
  assert formatted == ["x", "y"]


def test_argument_formatting_tuple(builder):
  """
  Scenario: [("x", "Tensor"), "dim"]
  """
  std_args = [("x", "Tensor"), "dim"]
  formatted = builder._format_args(std_args)
  assert formatted == ["x: Tensor", "dim"]


def test_argument_formatting_dict(builder):
  """
  Scenario: [{"name": "dim", "type": "int", "default": "-1"}]
  """
  std_args = [{"name": "dim", "type": "int", "default": "-1"}]
  formatted = builder._format_args(std_args)
  assert formatted == ["dim: int = -1"]


def test_missing_property_defaults(builder):
  """
  Verify safe handling of missing description/fields.
  """
  context = builder.build("EmptyOp", {})
  assert context["name"] == "EmptyOp"
  assert context["description"] == "No description available."
  assert context["args"] == []
  assert context["variants"] == []


def test_impl_type_classification_plugin(builder):
  """Verify Plugin detection."""
  var = {"requires_plugin": "my_hook"}
  assert builder._determine_impl_type(var) == "Plugin (my_hook)"


def test_impl_type_classification_macro(builder):
  """Verify Macro detection."""
  var = {"macro_template": "{x}*2"}
  assert builder._determine_impl_type(var) == "Macro '{x}*2'"


def test_impl_type_classification_infix(builder):
  """Verify Infix detection."""
  var = {"transformation_type": "infix", "operator": "+"}
  assert builder._determine_impl_type(var) == "Infix (+)"


def test_impl_type_classification_direct(builder):
  """Verify Direct Mapping."""
  var = {"api": "torch.abs"}
  assert builder._determine_impl_type(var) == "Direct Mapping"


def test_full_build_flow_with_adapter_logic(builder):
  """
  Verify end-to-end build combining all logic.
  Mocks `get_adapter` to test display name and URL resolution.
  """
  op_def = {
    "description": "Calculate abs.",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"requires_plugin": "magic"}, "unknown_fw": {"api": "foo"}},
  }

  # Mock Adapters for Torch and JAX
  mock_torch = MagicMock()
  mock_torch.display_name = "PyTorch"
  mock_torch.get_doc_url.return_value = "http://torch/abs"

  mock_jax = MagicMock()
  mock_jax.display_name = "JAX"
  # Plugins don't usually get doc URLs in the logic

  def get_adapter_side_effect(name):
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  # Mock priority order
  with patch("ml_switcheroo.utils.doc_context.get_framework_priority_order", return_value=["torch", "jax"]):
    with patch("ml_switcheroo.utils.doc_context.get_adapter", side_effect=get_adapter_side_effect):
      context = builder.build("Abs", op_def)

  assert context["name"] == "Abs"
  assert context["description"] == "Calculate abs."
  assert context["args"] == ["x"]

  assert len(context["variants"]) == 3

  # Verify Sorting (Torch first)
  v0 = context["variants"][0]
  assert v0["key"] == "torch"
  assert v0["framework"] == "PyTorch"
  assert v0["api"] == "torch.abs"
  assert v0["doc_url"] == "http://torch/abs"
  assert v0["implementation_type"] == "Direct Mapping"

  # Verify JAX (Plugin)
  v1 = context["variants"][1]
  assert v1["key"] == "jax"
  assert v1["implementation_type"] == "Plugin (magic)"
  # Logic suppresses URL for plugins
  assert v1["doc_url"] is None

  # Verify Unknown FW (Sorts last)
  v2 = context["variants"][2]
  assert v2["key"] == "unknown_fw"
  assert v2["framework"] == "unknown_fw"  # Fallback to key
