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
  assert render_node(convert_to_cst_literal(-5)) == "-5"
  assert render_node(convert_to_cst_literal(-3.14)) == "-3.14"

  class CustomObj:
    """Custom obj."""

    def __str__(self):
      """Str."""
      return "custom"

  assert render_node(convert_to_cst_literal(CustomObj())) == "'custom'"


def test_get_import_root():
  """Tests get_import_root."""
  from ml_switcheroo.tools.injector_fw.utils import get_import_root
  import libcst as cst

  assert get_import_root(cst.Name("torch")) == "torch"
  attr = cst.Attribute(value=cst.Name("scipy"), attr=cst.Name("special"))
  assert get_import_root(attr) == "scipy"
  assert get_import_root(cst.Integer("1")) == ""


def test_is_docstring():
  """Tests is_docstring."""
  from ml_switcheroo.tools.injector_fw.utils import is_docstring
  import libcst as cst

  # Not index 0
  assert is_docstring(cst.Name("test"), 1) is False

  # Correct format
  doc = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('"""Doc"""'))])
  assert is_docstring(doc, 0) is True

  # Wrong type
  wrong = cst.SimpleStatementLine(body=[cst.Pass()])
  assert is_docstring(wrong, 0) is False


def test_is_future_import():
  """Tests is_future_import."""
  from ml_switcheroo.tools.injector_fw.utils import is_future_import
  import libcst as cst

  # Future import
  tree = cst.parse_module("from __future__ import annotations")
  assert is_future_import(tree.body[0]) is True

  # Normal import
  tree = cst.parse_module("from os import path")
  assert is_future_import(tree.body[0]) is False

  # Not a SimpleStatementLine
  assert is_future_import(cst.Pass()) is False


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
