"""Test suite for `scripts/check_docstrings.py`."""

import ast
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_docstrings import (
  check_file,
  extract_parameter_names,
  find_stale_args,
  find_undocumented_class_attributes,
  has_non_none_return,
  has_uncaught_raise,
  is_arg_documented,
  is_generator_function,
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


def test_is_arg_documented_and_stale_args() -> None:
  """Verify strict `Args:` entry matching, underscore handling, missing `Args:` rejection, and stale detection."""
  doc = """Compute value using gamma in summary prose.

Args:
    alpha (int): First factor mentioning delta in prose.
        Continuation line for alpha.
    beta: Second factor.
    stale_param: No longer in signature.
"""
  assert is_arg_documented(doc, "alpha") is True
  assert is_arg_documented(doc, "_beta") is True
  assert is_arg_documented(doc, "gamma") is False
  assert is_arg_documented(doc, "delta") is False
  assert is_arg_documented(doc, "___") is False

  doc_no_section = "Compute value using alpha."
  assert is_arg_documented(doc_no_section, "alpha") is False
  assert find_stale_args(doc_no_section, ["alpha"]) == []

  empty_args_doc = chr(10).join(["Summary.", "", "Args:", "   ", "Returns:", "    int: Value."])
  assert find_stale_args(empty_args_doc, []) == []

  assert find_stale_args(doc, ["alpha", "_beta"]) == ["stale_param"]


def test_generator_and_uncaught_raise_detection() -> None:
  """Verify `is_generator_function` and `has_uncaught_raise` respect nested scopes and caught `try` blocks."""
  nl = chr(10)
  gen_fn = ast.parse(nl.join(["def g():", "  yield 1"])).body[0]
  assert isinstance(gen_fn, ast.FunctionDef)
  assert is_generator_function(gen_fn) is True

  nested_gen_fn = ast.parse(nl.join(["def outer():", "  def inner():", "    yield 1", "  return inner"])).body[0]
  assert isinstance(nested_gen_fn, ast.FunctionDef)
  assert is_generator_function(nested_gen_fn) is False

  caught_fn = ast.parse(
    nl.join(
      [
        "def safe():",
        "  def inner_raise():",
        "    raise KeyError('x')",
        "  try:",
        "    raise ValueError('caught')",
        "  except ValueError:",
        "    pass",
      ]
    )
  ).body[0]
  assert isinstance(caught_fn, ast.FunctionDef)
  assert has_uncaught_raise(caught_fn) is False

  try_finally_fn = ast.parse(
    nl.join(["def tf():", "  try:", "    raise RuntimeError('escapes')", "  finally:", "    pass"])
  ).body[0]
  assert isinstance(try_finally_fn, ast.FunctionDef)
  assert has_uncaught_raise(try_finally_fn) is True

  reraise_fn = ast.parse(nl.join(["def rr():", "  try:", "    pass", "  except ValueError:", "    raise"])).body[0]
  assert isinstance(reraise_fn, ast.FunctionDef)
  assert has_uncaught_raise(reraise_fn) is True


def test_find_undocumented_class_attributes() -> None:
  """Verify public `AnnAssign` checks support `Attributes:`, inline docstrings, `Field(description=...)`, and skip `_`/`model_config`."""
  code = chr(10).join(
    [
      "class SampleModel:",
      '  """Model docstring.',
      "",
      "  Attributes:",
      "      documented_attr: Documented in class header.",
      '  """',
      "  model_config = {}",
      "  model_config: dict[str, str] = {}",
      "  _private: int = 1",
      "  documented_attr: int",
      "  inline_attr: str",
      '  """Inline attribute documentation."""',
      "  field_attr: float = Field(default=1.0, description='Field doc')",
      "  missing_attr: bool",
    ]
  )
  cls_node = ast.parse(code).body[0]
  assert isinstance(cls_node, ast.ClassDef)
  doc = ast.get_docstring(cls_node) or ""
  assert find_undocumented_class_attributes(cls_node, doc) == ["missing_attr"]


def test_has_non_none_return_branches() -> None:
  """Verify `has_non_none_return` handles unannotated, `-> None`, and typed return nodes."""
  fn_no_ret = ast.parse("def f(): pass").body[0]
  assert isinstance(fn_no_ret, ast.FunctionDef)
  assert has_non_none_return(fn_no_ret) is False

  fn_const_none = ast.parse("def f() -> None: pass").body[0]
  assert isinstance(fn_const_none, ast.FunctionDef)
  assert has_non_none_return(fn_const_none) is False

  fn_name_none = ast.parse("def f(): pass").body[0]
  assert isinstance(fn_name_none, ast.FunctionDef)
  fn_name_none.returns = ast.Name(id="None", ctx=ast.Load())
  assert has_non_none_return(fn_name_none) is False

  fn_int = ast.parse("def f() -> int: return 1").body[0]
  assert isinstance(fn_int, ast.FunctionDef)
  assert has_non_none_return(fn_int) is True


def test_check_file_clean_and_violations(tmp_path: Path) -> None:
  """Verify `check_file` reports missing, placeholder, stale, generator, raises, and attribute violations.

  Args:
      tmp_path: Temporary directory fixture for writing Python modules.
  """
  prod_dir = tmp_path / "pkg"
  prod_dir.mkdir()
  clean_file = prod_dir / "clean.py"
  clean_file.write_text(
    chr(10).join(
      [
        '"""Valid module documentation."""',
        "import typing",
        "",
        "class MyService:",
        '  """Valid service class documentation.',
        "",
        "  Attributes:",
        "      factor: Multiplier factor.",
        '  """',
        "  factor: int = 2",
        "",
        "  def compute(self, value: int) -> int:",
        '    """Compute scaled value.',
        "",
        "    Args:",
        "        value: Input integer.",
        "",
        "    Returns:",
        "        Scaled integer.",
        "",
        "    Raises:",
        "        ValueError: If value is negative.",
        '    """',
        "    if value < 0:",
        "      raise ValueError('neg')",
        "    return value * self.factor",
        "",
        "  def iter_items(self) -> typing.Iterator[int]:",
        '    """Stream items.',
        "",
        "    Yields:",
        "        Next integer.",
        '    """',
        "    yield self.factor",
      ]
    ),
    encoding="utf-8",
  )
  assert check_file(clean_file) == []

  missing_file = prod_dir / "missing.py"
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

  placeholder_file = prod_dir / "placeholder.py"
  placeholder_file.write_text(
    chr(10).join(
      [
        '"""Module docstring."""',
        "import typing",
        "class DummyClass:",
        '  """Class doc."""',
        "  undoc_field: int = 1",
        "  def bad_fn(self, missing_arg: int) -> int:",
        '    """Docstring.',
        "",
        "    Args:",
        "        ghost_arg: Stale entry.",
        '    """',
        "    raise RuntimeError('boom')",
        "  def bad_gen(self) -> typing.Iterator[int]:",
        '    """Stream values.',
        "",
        "    Returns:",
        "        Wrong section for generator.",
        '    """',
        "    yield 1",
      ]
    ),
    encoding="utf-8",
  )
  p_violations = check_file(placeholder_file, check_args=True)
  assert any("Placeholder module docstring" in v for v in p_violations)
  assert any("Placeholder class docstring" in v for v in p_violations)
  assert any("Undocumented public attribute 'undoc_field'" in v for v in p_violations)
  assert any("Placeholder function docstring" in v for v in p_violations)
  assert any("Undocumented argument 'missing_arg'" in v for v in p_violations)
  assert any("Stale documented argument 'ghost_arg'" in v for v in p_violations)
  assert any("Missing Returns section in 'bad_fn'" in v for v in p_violations)
  assert any("Missing Raises section in 'bad_fn'" in v for v in p_violations)
  assert any("Missing Yields section in generator 'bad_gen'" in v for v in p_violations)

  test_file = prod_dir / "test_sample.py"
  test_file.write_text(
    chr(10).join(
      [
        '"""Valid test module docstring."""',
        "class HelperFixture:",
        '  """Helper class in test file."""',
        "  untracked_attr: int = 0",
        "def test_something(tmp_dir: Path) -> None:",
        '  """Verify something works with fixture."""',
        "  assert tmp_dir is not None",
      ]
    ),
    encoding="utf-8",
  )
  assert len(check_file(test_file)) == 1
  assert check_file(test_file, check_args=False) == []

  synthetic_file = prod_dir / "synthetic.py"
  synthetic_file.write_text(
    chr(10).join(
      [
        '"""Valid module docstring."""',
        "def fn_summary(x: int) -> int:",
        '  """Execute implementation detail.',
        "",
        "  Args:",
        "      x: Valid arg description.",
        "",
        "  Returns:",
        "      Valid return description.",
        '  """',
        "  return x",
        "",
        "def fn_args_ret(y: int) -> int:",
        '  """Compute valid summary.',
        "",
        "  Args:",
        "      y: Test argument or fixture.",
        "",
        "  Returns:",
        "      Computed test result or fixture value.",
        '  """',
        "  return y",
        "",
        "class FakeHelper:",
        '  """Test helper class FakeHelper."""',
        "",
        "def fn_synth_summary() -> None:",
        '  """Verifies the behavior of foo."""',
        "",
        "def fn_synth_arg(z: int) -> None:",
        '  """Validate integer input.',
        "",
        "  Args:",
        "      z: The z (int) for fn synth arg.",
        '  """',
        "",
        "def fn_whitespace_only() -> None:",
        '  """   """',
      ]
    ),
    encoding="utf-8",
  )
  syn_violations = check_file(synthetic_file)
  assert len(syn_violations) == 6


def test_iter_python_files_and_main(tmp_path: Path) -> None:
  """Verify file discovery and CLI exit codes.

  Args:
      tmp_path: Temporary directory fixture for test packages.
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
  """Verify executing `scripts.check_docstrings` as `__main__`.

  Args:
      tmp_path: Temporary directory fixture for target module.
  """
  good = tmp_path / "ok.py"
  good.write_text('"""Valid module docstring."""', encoding="utf-8")
  with patch.object(sys, "argv", ["scripts/check_docstrings.py", str(good)]):
    with pytest.raises(SystemExit) as exc:
      runpy.run_module("scripts.check_docstrings", run_name="__main__")
    assert exc.value.code == 0
