"""Tests for Latex parser coverage."""

import libcst as cst
from unittest.mock import patch
from ml_switcheroo.core.latex.parser import LatexParser
from ml_switcheroo.core.latex.nodes import LatexNode


def test_parse_arg_list_empty():
  """Test parsing empty arg list."""
  parser = LatexParser("")
  assert parser._parse_arg_list("   ") == []


def test_safe_value_node_ellipsis():
  """Test safe value node ellipsis."""
  parser = LatexParser("")
  node = parser._safe_value_node("...")
  assert isinstance(node, cst.Ellipsis)


def test_safe_value_node_fallback_name():
  """Test safe value node fallback name."""
  parser = LatexParser("")
  with patch("libcst.parse_expression", side_effect=cst.ParserSyntaxError("err", lines=[""], raw_line=0, raw_column=0)):
    node = parser._safe_value_node("valid_name")
    assert isinstance(node, cst.Name)
    assert node.value == "valid_name"


def test_create_call_simple_name():
  """Test create call simple name."""
  parser = LatexParser("")
  call = parser._create_call("SimpleFunc")
  assert isinstance(call.func, cst.Name)
  assert call.func.value == "SimpleFunc"


def test_create_call_config_arg():
  """Test create call config arg."""
  parser = LatexParser("")
  call = parser._create_call("Func", config={"arg_0": "val"})
  assert isinstance(call.args[0].value, cst.Name)
  assert call.args[0].value.value == "val"


def test_create_call_args_list_kwarg():
  """Test create call args list kwarg."""
  parser = LatexParser("")
  call = parser._create_call("Func", args_list=["k=v"])
  assert call.args[0].keyword.value == "k"
  assert call.args[0].value.value == "v"


def test_synthesize_class_fallback_op():
  """Test synthesize class fallback op."""
  parser = LatexParser("")

  class DummyNode(LatexNode):
    """Dummy."""

    def __init__(self):
      """Init."""
      self.node_id = "dummy_id"

    def to_latex(self):
      """To latex."""
      return ""

  dummy = DummyNode()
  class_def = parser._synthesize_class("Test", [], None, [dummy], None)
  # The body of the forward function should assign None to dummy_id
  fwd_func = class_def.body.body[1]
  assign = fwd_func.body.body[0].body[0]
  assert assign.targets[0].target.value == "dummy_id"
  assert assign.value.value == "None"
