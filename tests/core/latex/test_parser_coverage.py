"""Tests for Latex parser coverage."""

import typing
from unittest.mock import patch

import libcst as cst

from ml_switcheroo.core.latex.nodes import LatexNode
from ml_switcheroo.core.latex.parser import LatexParser


def test_parse_arg_list_empty() -> None:
  """Docstring."""
  parser = LatexParser("")
  assert parser._parse_arg_list("   ") == []


def test_safe_value_node_ellipsis() -> None:
  """Docstring."""
  parser = LatexParser("")
  node: typing.Any = parser._safe_value_node("...")
  assert isinstance(node, cst.Ellipsis)


def test_safe_value_node_fallback_name() -> None:
  """Docstring."""
  parser = LatexParser("")
  with patch("libcst.parse_expression", side_effect=cst.ParserSyntaxError("err", lines=[""], raw_line=0, raw_column=0)):
    node: typing.Any = parser._safe_value_node("valid_name")
    assert isinstance(node, cst.Name)
    assert node.value == "valid_name"


def test_create_call_simple_name() -> None:
  """Docstring."""
  parser = LatexParser("")
  call: typing.Any = parser._create_call("SimpleFunc")
  assert isinstance(call.func, cst.Name)
  assert call.func.value == "SimpleFunc"


def test_create_call_config_arg() -> None:
  """Docstring."""
  parser = LatexParser("")
  call: typing.Any = parser._create_call("Func", config={"arg_0": "val"})
  assert isinstance(call.args[0].value, cst.Name)
  assert call.args[0].value.value == "val"


def test_create_call_args_list_kwarg() -> None:
  """Docstring."""
  parser = LatexParser("")
  call: typing.Any = parser._create_call("Func", args_list=["k=v"])
  assert typing.cast(cst.Name, call.args[0].keyword).value == "k"  # type: ignore
  assert typing.cast(cst.Name, call.args[0].value).value == "v"


def test_synthesize_class_fallback_op() -> None:
  """Docstring."""
  parser = LatexParser("")

  class DummyNode(LatexNode):
    """Dummy."""

    def __init__(self) -> None:
      """Init."""
      self.node_id = "dummy_id"

    def to_latex(self) -> str:
      """To latex."""
      return ""

  dummy = DummyNode()
  class_def: typing.Any = parser._synthesize_class("Test", [], None, [dummy], None)  # type: ignore
  # The body of the forward function should assign None to dummy_id
  fwd_func: typing.Any = class_def.body.body[1]
  assign: typing.Any = fwd_func.body.body[0].body[0]
  assert assign.targets[0].target.value == "dummy_id"
  assert assign.value.value == "None"
