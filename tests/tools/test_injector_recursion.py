"""Test suite for the Injector Recursion module."""

import libcst as cst
from ml_switcheroo.tools.injector_fw import convert_to_cst_literal


def render_node(node: cst.CSTNode) -> str:
  """Renders node."""
  module = cst.parse_module("")
  return module.code_for_node(node)


def test_primitive_recursion():
  """Verifies the behavior of primitive recursion."""
  assert render_node(convert_to_cst_literal(1)) == "1"
  assert render_node(convert_to_cst_literal(1.5)) == "1.5"
  assert render_node(convert_to_cst_literal(True)) == "True"
  assert render_node(convert_to_cst_literal(None)) == "None"
  assert render_node(convert_to_cst_literal("foo")) == '"foo"'


def test_list_recursion():
  """Verifies the behavior of list recursion."""
  val = [1, 2, "a"]
  node = convert_to_cst_literal(val)
  code = render_node(node)
  assert code == '[1, 2, "a"]'


def test_tuple_recursion():
  """Verifies the behavior of tuple recursion."""
  val = (1, (2, 3))
  node = convert_to_cst_literal(val)
  code = render_node(node)
  clean = code.replace(" ", "")
  assert clean == "(1,(2,3))"


def test_dict_recursion():
  """Verifies the behavior of dictionary recursion."""
  val = {"alpha": 0.5, "dims": (1, 2), "flag": True}
  node = convert_to_cst_literal(val)
  code = render_node(node)
  clean = code.replace(" ", "")
  assert '"alpha":0.5' in clean
  assert '"dims":(1,2)' in clean
  assert '"flag":True' in clean


def test_deep_nesting():
  """Verifies the behavior of deep nesting."""
  val = [{"a": [1, 2]}, (None,)]
  node = convert_to_cst_literal(val)
  code = render_node(node)
  clean = code.replace(" ", "")
  assert '[{"a":[1,2]},(None,)]' == clean
