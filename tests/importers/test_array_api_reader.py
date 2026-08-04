"""Unit test suite for the Array API Standard specification importer.

This module validates the functionality of the `ArrayApiSpecImporter` class,
which is responsible for parsing Python stub files (*.py) from the official
Array API standard repository. It covers folder parsing, relative path error handling,
syntax error recovery, function and constant signature extraction, type hint processing,
and docstring cleaning.
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter


@pytest.fixture
def importer():
  """Provides a clean importer instance for testing.

  Returns:
      ArrayApiSpecImporter: An instance of `ArrayApiSpecImporter` to be used in test cases.
  """
  return ArrayApiSpecImporter()


def test_parse_folder_no_files(importer, tmp_path):
  """Verifies that parsing an empty folder returns an empty dictionary.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  assert importer.parse_folder(tmp_path) == {}


def test_parse_folder_with_files(importer, tmp_path):
  """Verifies that parsing a directory containing valid stub files extracts defined functions.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  file1 = tmp_path / "test1.py"
  file1.write_text("def my_func(x: int):\n    '''Docstring'''\n    pass")
  result = importer.parse_folder(tmp_path)
  assert "my_func" in result


def test_parse_stubs_skip_private(importer, tmp_path):
  """Verifies that the importer skips private files but includes __init__.py files.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  file1 = tmp_path / "_private.py"
  file1.write_text("def func(): pass")
  file2 = tmp_path / "__init__.py"
  file2.write_text("def func2(): pass")
  result = importer._parse_stubs([file1, file2], tmp_path)
  assert "func" not in result
  assert "func2" in result


def test_parse_stubs_relative_path_error(importer, tmp_path):
  """Verifies correct handling of relative path errors when calculation fails.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  Path("/some/outside/path.py")
  file1_mock = MagicMock(spec=Path)
  file1_mock.name = "path.py"
  file1_mock.relative_to.side_effect = ValueError
  file1_mock.read_text.return_value = "def func(): pass"
  result = importer._parse_stubs([file1_mock], tmp_path)
  assert result["func"]["from"] == "path.py"


def test_parse_stubs_parse_error(importer, tmp_path):
  """Verifies that files with invalid Python syntax are gracefully ignored.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  file1 = tmp_path / "bad.py"
  file1.write_text("def bad_syntax(")
  result = importer._parse_stubs([file1], tmp_path)
  assert result == {}


def test_parse_stubs_function_parsing(importer, tmp_path):
  """Verifies that function signatures, descriptions, and standard arguments are correctly extracted.

  This also tests that private helper functions are ignored while special/magic methods are included.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  code = '\ndef valid_func(x: int, /, y: float, *, z: str):\n    """\n    My function summary.\n\n    Detailed description.\n    """\n    pass\n\ndef _private_helper():\n    pass\n\ndef __magic_method__():\n    pass\n'
  file1 = tmp_path / "funcs.py"
  file1.write_text(code)
  result = importer._parse_stubs([file1], tmp_path)
  assert "valid_func" in result
  assert result["valid_func"]["description"] == "My function summary."
  assert result["valid_func"]["std_args"] == [("x", "int"), ("y", "float"), ("z", "str")]
  assert "_private_helper" not in result
  assert "__magic_method__" in result


def test_parse_stubs_constant_parsing(importer, tmp_path):
  """Verifies that constants and their lookahead docstrings are properly parsed and extracted.

  This also tests that private constants are skipped.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.
      tmp_path (Path): A temporary path directory provided by pytest.

  Returns:
      None
  """
  code = '\nE = 2.718\n"""Euler\'s constant."""\n\nPI: float = 3.14\n"""Pi."""\n\n_PRIVATE_CONST = 1\n'
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
  """Verifies parsing of various type annotations into standard string representations.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.

  Returns:
      None
  """
  assert importer._parse_annotation(None) == "Any"
  assert importer._parse_annotation(ast.parse("x: int").body[0].annotation) == "int"
  assert importer._parse_annotation(ast.parse("x: 'MyType'").body[0].annotation) == "MyType"
  assert importer._parse_annotation(ast.parse("x: Optional[int]").body[0].annotation) == "Optional[int]"
  assert importer._parse_annotation(ast.parse("x: Tuple[int, str]").body[0].annotation) == "Tuple[int, str]"
  fake_sub = MagicMock(spec=ast.Subscript)
  del fake_sub.slice
  fake_sub.value = ast.Name(id="List", ctx=ast.Load())
  assert importer._parse_annotation(fake_sub) == "List"
  assert importer._parse_annotation(ast.parse("x: int | float").body[0].annotation) == "int | float"
  assert importer._parse_annotation(ast.parse("x: types.NoneType").body[0].annotation) == "types.NoneType"
  assert importer._parse_annotation(ast.parse("x: lambda: None").body[0].annotation) == "Any"


def test_get_assignment_name(importer):
  """Verifies the retrieval of assignment names from AST assignment and annotated assignment nodes.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.

  Returns:
      None
  """
  assert importer._get_assignment_name(ast.parse("x = 1").body[0]) == "x"
  assert importer._get_assignment_name(ast.parse("x.y = 1").body[0]) is None
  assert importer._get_assignment_name(ast.parse("x: int = 1").body[0]) == "x"
  assert importer._get_assignment_name(ast.parse("x.y: int = 1").body[0]) is None
  assert importer._get_assignment_name(ast.parse("pass").body[0]) is None


def test_clean_docstring(importer):
  """Verifies the cleaning, extraction, and formatting of multi-line and single-line docstrings.

  Args:
      importer (ArrayApiSpecImporter): The spec importer fixture.

  Returns:
      None
  """
  assert importer._clean_docstring(None) == ""
  assert importer._clean_docstring("  \nSingle line summary.  \n\nDetailed doc.") == "Single line summary."
  assert importer._clean_docstring("Line 1.\nLine 2.\n\nLine 3.") == "Line 1. Line 2."
