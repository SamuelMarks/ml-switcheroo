import ast
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter


@pytest.fixture
def importer():
  return ArrayApiSpecImporter()


def test_parse_folder_no_files(importer, tmp_path):
  assert importer.parse_folder(tmp_path) == {}


def test_parse_folder_with_files(importer, tmp_path):
  file1 = tmp_path / "test1.py"
  file1.write_text("def my_func(x: int):\n    '''Docstring'''\n    pass")

  result = importer.parse_folder(tmp_path)
  assert "my_func" in result


def test_parse_stubs_skip_private(importer, tmp_path):
  file1 = tmp_path / "_private.py"
  file1.write_text("def func(): pass")
  file2 = tmp_path / "__init__.py"
  file2.write_text("def func2(): pass")

  result = importer._parse_stubs([file1, file2], tmp_path)
  assert "func" not in result
  assert "func2" in result


def test_parse_stubs_relative_path_error(importer, tmp_path):
  file1 = Path("/some/outside/path.py")

  # Mocking the read_text to avoid FileNotFoundError
  file1_mock = MagicMock(spec=Path)
  file1_mock.name = "path.py"
  file1_mock.relative_to.side_effect = ValueError
  file1_mock.read_text.return_value = "def func(): pass"

  result = importer._parse_stubs([file1_mock], tmp_path)
  assert result["func"]["from"] == "path.py"


def test_parse_stubs_parse_error(importer, tmp_path):
  file1 = tmp_path / "bad.py"
  file1.write_text("def bad_syntax(")

  result = importer._parse_stubs([file1], tmp_path)
  assert result == {}


def test_parse_stubs_function_parsing(importer, tmp_path):
  code = '''
def valid_func(x: int, /, y: float, *, z: str):
    """
    My function summary.
    
    Detailed description.
    """
    pass

def _private_helper():
    pass

def __magic_method__():
    pass
'''
  file1 = tmp_path / "funcs.py"
  file1.write_text(code)

  result = importer._parse_stubs([file1], tmp_path)

  assert "valid_func" in result
  assert result["valid_func"]["description"] == "My function summary."
  assert result["valid_func"]["std_args"] == [("x", "int"), ("y", "float"), ("z", "str")]

  assert "_private_helper" not in result
  assert "__magic_method__" in result


def test_parse_stubs_constant_parsing(importer, tmp_path):
  code = '''
E = 2.718
"""Euler's constant."""

PI: float = 3.14
"""Pi."""

_PRIVATE_CONST = 1
'''
  file1 = tmp_path / "consts.py"
  file1.write_text(code)

  result = importer._parse_stubs([file1], tmp_path)

  assert "E" in result
  assert result["E"]["description"] == "Euler's constant."
  assert result["E"]["std_args"] == []

  assert "PI" in result
  assert result["PI"]["description"] == "Pi."

  assert "_PRIVATE_CONST" not in result


def test_parse_annotation(importer):
  # None
  assert importer._parse_annotation(None) == "Any"

  # ast.Name
  assert importer._parse_annotation(ast.parse("x: int").body[0].annotation) == "int"

  # ast.Constant
  assert importer._parse_annotation(ast.parse("x: 'MyType'").body[0].annotation) == "MyType"

  # ast.Subscript with simple slice
  assert importer._parse_annotation(ast.parse("x: Optional[int]").body[0].annotation) == "Optional[int]"

  # ast.Subscript with Tuple
  assert importer._parse_annotation(ast.parse("x: Tuple[int, str]").body[0].annotation) == "Tuple[int, str]"

  # ast.Subscript without slice attribute (Python < 3.9 representation simulation)
  fake_sub = MagicMock(spec=ast.Subscript)
  del fake_sub.slice
  fake_sub.value = ast.Name(id="List", ctx=ast.Load())
  assert importer._parse_annotation(fake_sub) == "List"

  # ast.BinOp (Union with BitOr)
  assert importer._parse_annotation(ast.parse("x: int | float").body[0].annotation) == "int | float"

  # ast.Attribute
  assert importer._parse_annotation(ast.parse("x: types.NoneType").body[0].annotation) == "types.NoneType"

  # Fallback
  assert importer._parse_annotation(ast.parse("x: lambda: None").body[0].annotation) == "Any"


def test_get_assignment_name(importer):
  # ast.Assign
  assert importer._get_assignment_name(ast.parse("x = 1").body[0]) == "x"
  assert importer._get_assignment_name(ast.parse("x.y = 1").body[0]) is None

  # ast.AnnAssign
  assert importer._get_assignment_name(ast.parse("x: int = 1").body[0]) == "x"
  assert importer._get_assignment_name(ast.parse("x.y: int = 1").body[0]) is None

  # Fallback
  assert importer._get_assignment_name(ast.parse("pass").body[0]) is None


def test_clean_docstring(importer):
  assert importer._clean_docstring(None) == ""
  assert importer._clean_docstring("  \nSingle line summary.  \n\nDetailed doc.") == "Single line summary."
  assert importer._clean_docstring("Line 1.\nLine 2.\n\nLine 3.") == "Line 1. Line 2."
