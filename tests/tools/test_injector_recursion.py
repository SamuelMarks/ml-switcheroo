"""Test suite for the Injector Recursion module."""

from typing import Any, Dict, List, Tuple

import libcst as cst

from ml_switcheroo.tools.injector_fw import convert_to_cst_literal


def render_node(node: cst.CSTNode) -> str:
  """Renders node."""
  module: cst.Module = cst.parse_module("")
  return getattr(module, "code_for_node")(node)


def test_primitive_recursion() -> None:
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

    def __str__(self) -> str:
      """Str."""
      return "custom"

  assert render_node(convert_to_cst_literal(CustomObj())) == "'custom'"


def test_get_import_root() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import get_import_root

  assert get_import_root(cst.Name("torch")) == "torch"
  attr: cst.Attribute = cst.Attribute(value=cst.Name("scipy"), attr=cst.Name("special"))
  assert get_import_root(attr) == "scipy"
  assert get_import_root(cst.Integer("1")) == ""


def test_is_docstring() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import is_docstring

  # Not index 0
  assert is_docstring(cst.Name("test"), 1) is False

  # Correct format
  doc: cst.SimpleStatementLine = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('"""Doc"""'))])
  assert is_docstring(doc, 0) is True

  # Wrong type
  wrong: cst.SimpleStatementLine = cst.SimpleStatementLine(body=[cst.Pass()])
  assert is_docstring(wrong, 0) is False


def test_is_future_import() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.tools.injector_fw.utils import is_future_import

  # Future import
  tree: cst.Module = cst.parse_module("from __future__ import annotations")
  assert is_future_import(getattr(tree, "body")[0]) is True

  # Normal import
  tree2: cst.Module = cst.parse_module("from os import path")
  assert is_future_import(getattr(tree2, "body")[0]) is False

  # Not a SimpleStatementLine
  assert is_future_import(cst.Pass()) is False


def test_list_recursion() -> None:
  """Verifies the behavior of list recursion."""
  val: List[Any] = [1, 2, "a"]
  node: cst.CSTNode = convert_to_cst_literal(val)
  code: str = render_node(node)
  assert code == '[1, 2, "a"]'


def test_tuple_recursion() -> None:
  """Verifies the behavior of tuple recursion."""
  val: Tuple[Any, ...] = (1, (2, 3))
  node: cst.CSTNode = convert_to_cst_literal(val)
  code: str = render_node(node)
  clean: str = code.replace(" ", "")
  assert clean == "(1,(2,3))"


def test_dict_recursion() -> None:
  """Verifies the behavior of dictionary recursion."""
  val: Dict[str, Any] = {"alpha": 0.5, "dims": (1, 2), "flag": True}
  node: cst.CSTNode = convert_to_cst_literal(val)
  code: str = render_node(node)
  clean: str = code.replace(" ", "")
  assert '"alpha":0.5' in clean
  assert '"dims":(1,2)' in clean
  assert '"flag":True' in clean


def test_deep_nesting() -> None:
  """Verifies the behavior of deep nesting."""
  val: List[Any] = [{"a": [1, 2]}, (None,)]
  node: cst.CSTNode = convert_to_cst_literal(val)
  code: str = render_node(node)
  clean: str = code.replace(" ", "")
  assert '[{"a":[1,2]},(None,)]' == clean
