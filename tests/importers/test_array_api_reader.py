"""Test module."""

import ast
import typing
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter


def test_array_api_reader(tmp_path: Path) -> None:
  """Docstring."""
  # Create some dummy .py stubs
  (tmp_path / "valid.py").write_text('''
e = 2.718
"""Euler's number"""

PI: float = 3.14
"""Pi"""

def add(x: Array, y: Optional[Array]) -> Array:
    """Adds two arrays."""
    pass

def sub(x: Array, /, y: Array, *, out: Optional[Array] = None) -> Array:
    """Subtracts arrays."""
    pass

def _private(x):
    pass

_hidden = 1
''')

  (tmp_path / "_types.py").write_text("""
def should_skip():
    pass
""")

  (tmp_path / "__init__.py").write_text("""
def re_exported():
    pass
""")

  (tmp_path / "invalid.py").write_text("""
def broken(
""")

  importer = ArrayApiSpecImporter()

  # Empty dir
  empty_dir = tmp_path / "empty"
  empty_dir.mkdir()
  res_empty = importer.parse_folder(empty_dir)
  assert res_empty == {}

  # Valid dir
  res = importer.parse_folder(tmp_path)

  assert "add" in res
  assert res["add"]["description"] == "Adds two arrays."
  assert res["add"]["std_args"] == [("x", "Array"), ("y", "Optional[Array]")]

  assert "sub" in res
  assert res["sub"]["std_args"] == [("x", "Array"), ("y", "Array"), ("out", "Optional[Array]")]

  assert "e" in res
  assert res["e"]["description"] == "Euler's number"

  assert "PI" in res
  assert res["PI"]["description"] == "Pi"

  assert "re_exported" in res
  assert "_private" not in res
  assert "_hidden" not in res
  assert "should_skip" not in res


def test_parse_annotation() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()

  # Test None
  assert importer._parse_annotation(None) == "Any"

  # Test Name
  node_name = ast.Name(id="int", ctx=ast.Load())
  assert importer._parse_annotation(node_name) == "int"

  # Test Constant
  node_const = ast.Constant(value="str")
  assert importer._parse_annotation(node_const) == "str"

  # Test Subscript
  node_sub = ast.Subscript(
    value=ast.Name(id="Optional", ctx=ast.Load()), slice=ast.Name(id="int", ctx=ast.Load()), ctx=ast.Load()
  )
  assert importer._parse_annotation(node_sub) == "Optional[int]"

  # Test Subscript with Tuple
  node_sub_tuple = ast.Subscript(
    value=ast.Name(id="Tuple", ctx=ast.Load()),
    slice=ast.Tuple(elts=[ast.Name(id="int", ctx=ast.Load()), ast.Name(id="float", ctx=ast.Load())], ctx=ast.Load()),
    ctx=ast.Load(),
  )
  assert importer._parse_annotation(node_sub_tuple) == "Tuple[int, float]"

  # Test BinOp (Union)
  node_binop = ast.BinOp(
    left=ast.Name(id="int", ctx=ast.Load()), op=ast.BitOr(), right=ast.Name(id="float", ctx=ast.Load())
  )
  assert importer._parse_annotation(node_binop) == "int | float"

  # Test BinOp fallback
  node_binop2 = ast.BinOp(
    left=ast.Name(id="int", ctx=ast.Load()), op=ast.Add(), right=ast.Name(id="float", ctx=ast.Load())
  )
  assert importer._parse_annotation(node_binop2) == "Any"

  # Test Attribute
  node_attr = ast.Attribute(value=ast.Name(id="types", ctx=ast.Load()), attr="NoneType", ctx=ast.Load())
  assert importer._parse_annotation(node_attr) == "types.NoneType"

  # Test Fallback
  node_fallback = ast.Pass()
  assert importer._parse_annotation(node_fallback) == "Any"


def test_get_assignment_name() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()

  # Test Assign
  node_assign = ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=1))
  assert importer._get_assignment_name(node_assign) == "x"

  node_assign2 = ast.Assign(targets=[ast.Attribute(value=ast.Name(id="self"), attr="x")], value=ast.Constant(value=1))
  assert importer._get_assignment_name(node_assign2) is None

  # Test AnnAssign
  node_annassign = ast.AnnAssign(
    target=ast.Name(id="y", ctx=ast.Store()), annotation=ast.Name(id="int"), value=ast.Constant(value=1), simple=1
  )
  assert importer._get_assignment_name(node_annassign) == "y"

  node_annassign2 = ast.AnnAssign(
    target=ast.Attribute(value=ast.Name(id="self"), attr="y"),
    annotation=ast.Name(id="int"),
    value=ast.Constant(value=1),
    simple=0,
  )
  assert importer._get_assignment_name(node_annassign2) is None

  # Test fallback
  node_fallback = ast.Pass()
  assert importer._get_assignment_name(node_fallback) is None


def test_clean_docstring() -> None:
  """Docstring."""
  importer = ArrayApiSpecImporter()
  assert importer._clean_docstring(None) == ""
  assert importer._clean_docstring("   ") == ""
  assert importer._clean_docstring("Line 1\nLine 2\n\nLine 3") == "Line 1 Line 2"


# --- Merged from test_array_api_reader_extra.py ---


def test_array_api_reader_relative_to_value_error(tmp_path: Path) -> None:
  """Docstring."""
  reader = ArrayApiSpecImporter()

  file_path: Path = tmp_path / "test.py"
  file_path.write_text("def foo(): pass")

  with patch("pathlib.Path.relative_to", side_effect=ValueError):
    res: dict[str, typing.Any] = reader._parse_stubs([file_path], tmp_path)

  assert "foo" in res
  assert res["foo"]["from"] == "test.py"


def test_array_api_reader_subscript_no_slice() -> None:
  """Docstring."""
  reader = ArrayApiSpecImporter()
  # Mock Subscript without slice
  node = ast.Subscript(value=ast.Name(id="List"), slice=ast.Name(id="Any"))
  if hasattr(node, "slice"):
    del node.slice
  assert reader._parse_annotation(node) == "List"


def test_array_api_reader_constants_branches(tmp_path: Path) -> None:
  """Test edge cases for constants docstring extraction."""
  (tmp_path / "consts.py").write_text("""
CONST_NO_DOC = 1
def not_doc(): pass

CONST_INT_EXPR = 2
42

CONST_LAST = 3
""")
  importer = ArrayApiSpecImporter()
  res = importer.parse_folder(tmp_path)
  assert res["CONST_NO_DOC"]["description"] == "Constant: CONST_NO_DOC"
  assert res["CONST_INT_EXPR"]["description"] == "Constant: CONST_INT_EXPR"
  assert res["CONST_LAST"]["description"] == "Constant: CONST_LAST"
