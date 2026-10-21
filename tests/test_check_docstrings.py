"""Test suite for scripts/check_docstrings.py."""

from __future__ import annotations


import ast
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_docstrings import (
  check_file,
  extract_parameter_names,
  is_arg_documented,
  is_test_path,
  iter_python_files,
  main,
)


def test_is_test_path() -> None:
  """Verify detection of test paths vs production paths."""
  assert is_test_path(Path("tests/test_foo.py")) is True
  assert is_test_path(Path("src/test_helper.py")) is True
  assert is_test_path(Path("src/conftest.py")) is True
  assert is_test_path(Path("src/ml_switcheroo/engine.py")) is False


def test_extract_parameter_names() -> None:
  """Verify extraction of positional-only, standard, kw-only, vararg, and kwarg parameters."""
  code = """def fn(self, cls, a, /, b, *args, c=1, **kwargs):
    pass
"""
  tree = ast.parse(code)
  fn_node = tree.body[0]
  assert isinstance(fn_node, ast.FunctionDef)
  assert extract_parameter_names(fn_node) == ["a", "b", "c", "args", "kwargs"]

  code_simple = """def fn_simple(self, x):
    pass
"""
  tree_simple = ast.parse(code_simple)
  fn_simple = tree_simple.body[0]
  assert isinstance(fn_simple, ast.FunctionDef)
  assert extract_parameter_names(fn_simple) == ["x"]


def test_is_arg_documented() -> None:
  """Verify argument matching with exact names and leading underscores."""
  doc = """Compute value.

Args:
    alpha: First factor.
    beta: Second factor.
"""
  assert is_arg_documented(doc, "alpha") is True
  assert is_arg_documented(doc, "_beta") is True
  assert is_arg_documented(doc, "gamma") is False
  assert is_arg_documented(doc, "___") is False

  doc_no_section = "Compute value using alpha."
  assert is_arg_documented(doc_no_section, "alpha") is True


def test_has_non_none_return_branches() -> None:
  """Verify `has_non_none_return` handles unannotated, `-> None`, and typed return nodes."""
  from scripts.check_docstrings import has_non_none_return

  fn_no_ret = ast.parse("def f(): pass").body[0]
  assert isinstance(fn_no_ret, ast.FunctionDef)
  assert has_non_none_return(fn_no_ret) is False

  fn_const_none = ast.parse("def f() -> None: pass").body[0]
  assert isinstance(fn_const_none, ast.FunctionDef)
  assert has_non_none_return(fn_const_none) is False

  fn_name_none = getattr(
    {
      "v": getattr({"v": getattr({"v": ast}, "__getitem__")("v").FunctionDef}, "__getitem__")("v")(
        name="f",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Pass()],
        decorator_list=[],
        returns=ast.Name(id="None", ctx=ast.Load()),
      )
    },
    "__getitem__",
  )("v")
  assert has_non_none_return(fn_name_none) is False

  fn_int = ast.parse("def f() -> int: return 1").body[0]
  assert isinstance(fn_int, ast.FunctionDef)
  assert has_non_none_return(fn_int) is True


def test_check_file_clean_and_violations(tmp_path: Path) -> None:
  """Verify check_file reports missing, placeholder, and undocumented arg violations.

  Args:
      tmp_path: Test argument or fixture.
  """
  clean_file = tmp_path / "clean.py"
  clean_file.write_text(
    chr(10).join(
      [
        '"""Valid module documentation."""',
        "",
        "class MyService:",
        '  """Valid service class documentation."""',
        "  def compute(self, value: int) -> int:",
        '    """Compute doubled value.',
        "",
        "    Args:",
        "        value: Input integer.",
        "",
        "    Returns:",
        "        Doubled integer.",
        '    """',
        "    return value * 2",
      ]
    ),
    encoding="utf-8",
  )
  assert check_file(clean_file) == []

  missing_file = tmp_path / "missing.py"
  missing_file.write_text(
    chr(10).join(
      [
        "class NoDocClass:",
        "  def no_doc_fn(self, x: int) -> int:",
        "    return x",
      ]
    ),
    encoding="utf-8",
  )
  violations = check_file(missing_file)
  assert len(violations) == 3
  assert any("Missing module docstring" in v for v in violations)
  assert any("Missing class docstring for 'NoDocClass'" in v for v in violations)
  assert any("Missing function docstring for 'no_doc_fn'" in v for v in violations)

  placeholder_file = tmp_path / "placeholder.py"
  placeholder_file.write_text(
    chr(10).join(
      [
        '"""Module docstring."""',
        "class DummyClass:",
        '  """Class doc."""',
        "  def bad_fn(self, missing_arg: int) -> int:",
        '    """Docstring."""',
        "    return missing_arg",
      ]
    ),
    encoding="utf-8",
  )
  p_violations = check_file(placeholder_file, check_args=True)
  assert len(p_violations) == 5
  assert any("Placeholder module docstring" in v for v in p_violations)
  assert any("Placeholder class docstring" in v for v in p_violations)
  assert any("Placeholder function docstring" in v for v in p_violations)
  assert any("Undocumented argument 'missing_arg'" in v for v in p_violations)
  assert any("Missing Returns section in 'bad_fn'" in v for v in p_violations)

  # Test file also enforces Args: by default unless check_args=False
  test_file = tmp_path / "test_sample.py"
  test_file.write_text(
    chr(10).join(
      [
        '"""Valid test module docstring."""',
        "def test_something(tmp_path: object) -> None:",
        '  """Verify something works with fixture."""',
        "  assert tmp_path is not None",
      ]
    ),
    encoding="utf-8",
  )
  assert len(check_file(test_file)) == 1
  assert check_file(test_file, check_args=False) == []


def test_iter_python_files_and_main(tmp_path: Path) -> None:
  """Verify file discovery and CLI exit codes.

  Args:
      tmp_path: Test argument or fixture.
  """
  sub = tmp_path / "pkg"
  sub.mkdir()
  good = sub / "good.py"
  good.write_text('"""Good module docstring."""', encoding="utf-8")
  non_py = sub / "notes.txt"
  non_py.write_text("ignore me", encoding="utf-8")

  discovered = iter_python_files([good, sub, non_py, tmp_path / "nonexistent"])
  assert discovered == [good, good]

  assert main([str(good)]) == 0

  bad = sub / "bad.py"
  bad.write_text("x = 1", encoding="utf-8")
  assert main([str(sub)]) == 1


def test_check_docstrings_dunder_main(tmp_path: Path) -> None:
  """Verify executing scripts.check_docstrings as __main__.

  Args:
      tmp_path: Test argument or fixture.
  """
  good = tmp_path / "ok.py"
  good.write_text('"""Valid module docstring."""', encoding="utf-8")
  with patch.object(sys, "argv", ["scripts/check_docstrings.py", str(good)]):
    with pytest.raises(SystemExit) as exc:
      runpy.run_module("scripts.check_docstrings", run_name="__main__")
    assert exc.value.code == 0
