"""
Tests for Type-Aware Spec Reader (Feature 027).

Verifies that:
1. Basic types (int, float) are extracted.
2. Complex types (Optional[int], int | float) are parsed to strings.
3. Arguments structure is now List[Tuple[name, type]].
"""

import ast
import pytest
from ml_switcheroo.importers.spec_reader import ArrayApiSpecImporter


@pytest.fixture
def importer():
  return ArrayApiSpecImporter()


def test_parse_simple_types(importer):
  """Verify 'x: int' -> 'int'."""
  code = "def func(x: int, y: float): pass"
  tree = ast.parse(code)
  func_node = tree.body[0]

  args = importer._extract_args(func_node.args)

  assert args == [("x", "int"), ("y", "float")]


def test_parse_complex_subscripts(importer):
  """Verify 'x: Optional[Tuple[int, int]]'."""
  code = "def shape(x: Optional[Tuple[int, int]]): pass"
  tree = ast.parse(code)
  func_node = tree.body[0]

  args = importer._extract_args(func_node.args)
  # The output string format depends on _parse_annotation implementation
  # Tuple[int, int] -> Tuple[int, int]
  assert "Optional" in args[0][1]
  assert "Tuple" in args[0][1]
  assert "int" in args[0][1]


def test_parse_union_syntax(importer):
  """Verify 'x: int | float' (Python 3.10+ style)."""
  code = "def add(x: int | float): pass"
  tree = ast.parse(code)
  func_node = tree.body[0]

  args = importer._extract_args(func_node.args)

  assert args[0][1] == "int | float"


def test_parse_folder_integration(importer, tmp_path):
  """
  Verify integration flow writes types to JSON structure in memory.
  """
  stub_file = tmp_path / "math.py"
  stub_file.write_text("def abs(x: Array) -> Array: ...")

  semantics = importer._parse_stubs([stub_file], tmp_path)

  assert "abs" in semantics
  std_args = semantics["abs"]["std_args"]

  # Structure check: List of tuples
  assert len(std_args) == 1
  assert std_args[0] == ("x", "Array")


def test_fallback_any(importer):
  """Verify missing annotations default to 'Any'."""
  code = "def untyped(x): pass"
  tree = ast.parse(code)
  func_node = tree.body[0]

  args = importer._extract_args(func_node.args)
  assert args[0] == ("x", "Any")
