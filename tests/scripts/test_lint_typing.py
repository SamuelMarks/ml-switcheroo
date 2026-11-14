"""Tests for `scripts/lint_typing.py`."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.lint_typing import check_source, collect_python_files, main


def test_lint_typing_clean_and_violations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verify `check_source`, `collect_python_files`, and `main` on clean and violating code.

  Args:
      tmp_path: Temporary directory path fixture.
      capsys: Pytest output capture fixture.
  """
  nl = chr(10)
  clean_file = tmp_path / "clean.py"
  clean_code = nl.join(
    [
      '"""Clean module."""',
      "import collections",
      "from collections import deque",
      "from typing import Sequence",
      "from builtins import int",
      "from operator import add",
      "from os import path",
      "x: int = 1",
      "d = collections.OrderedDict()",
      "r = ([1][0]).bit_length()",
      "u = ([1][0]).numerator.real",
      "p = path.sep.upper()",
      "class S:",
      "  def __init__(self) -> None:",
      "    super().__setattr__('a', 1)",
      "  async def compute(self) -> int:",
      "    return 1",
    ]
  )
  clean_file.write_text(f"{clean_code}{nl}", encoding="utf-8")
  non_py = tmp_path / "notes.txt"
  non_py.write_text("hello", encoding="utf-8")
  assert check_source(clean_file.read_text(encoding="utf-8"), clean_file) == []
  assert collect_python_files([clean_file, non_py, tmp_path]) == [clean_file]
  assert main([str(clean_file)]) == 0

  bad_code = nl.join(
    [
      '"""Bad module."""',
      "from typing import Any, cast as c_alias, Hashable",
      "from builtins import delattr, getattr as ga_alias",
      "from operator import itemgetter, attrgetter as ag_alias",
      "from collections import abc as cabc",
      "import collections.abc",
      "import operator as op",
      "import typing as t",
      "import builtins as b",
      "x: Any = 1",
      "h: Hashable = 2",
      "h2: cabc.Hashable = 3",
      "h3: collections.abc.Hashable = 4",
      "o: object = object()",
      "y: t.Any = t.cast(int, x)",
      "y2 = c_alias(int, x)",
      "z = b.getattr(x, 'real')",
      "z2 = ga_alias(x, 'real')",
      "g = op.attrgetter('real')(x)",
      "ig = ag_alias('real')(x)",
      "m = __import__('sys')",
      "c = compile('1', '<s>', 'eval')",
      "object.__setattr__(x, 'a', 1)",
      "type.__setattr__(int, 'a', 1)",
      "u = x.__add__(1)",
      "v = x.__call__()",
      "class C:",
      "  def __init__(self) -> None:",
      "    super().__init__()",
      "  def __getattr__(self, name: str) -> int:",
      "    return 0",
      "  async def __getattribute__(self, name: str) -> int:",
      "    return 0",
      "w = hasattr(x, 'real')  # type: ignore",
    ]
  )
  bad_file = tmp_path / "bad.py"
  bad_file.write_text(bad_code, encoding="utf-8")
  violations = check_source(bad_code, bad_file)
  assert any("async def __getattribute__" in v for v in violations)
  assert any("collections.abc.Hashable" in v for v in violations)
  assert any("Forbidden reference 'typing.Any'" in v for v in violations)
  assert any("Forbidden builtin reference 'builtins.getattr'" in v for v in violations)
  assert any("Forbidden operator reflection reference 'operator.attrgetter'" in v for v in violations)
  assert any("c_alias()" in v for v in violations)
  assert main([str(bad_file)]) == 1
  captured = capsys.readouterr()
  assert "Forbidden" in captured.err

  prod_dir = tmp_path / "src" / "pkg"
  prod_dir.mkdir(parents=True)
  prod_mock_file = prod_dir / "mod.py"
  prod_mock_code = nl.join(
    [
      '"""Prod module."""',
      "import inspect as insp",
      "import unittest.mock",
      "from pathlib import Path",
      "from inspect import signature, getattr_static as gs",
      "from unittest import TestCase, mock",
      "from unittest.mock import MagicMock",
      "x = insp.getattr_static",
      "y = gs(TestCase, '__doc__')",
    ]
  )
  prod_mock_file.write_text(prod_mock_code, encoding="utf-8")
  prod_violations = check_source(prod_mock_code, prod_mock_file)
  assert len(prod_violations) == 7
  assert any("inspect.getattr_static" in v for v in prod_violations)
  assert any("Forbidden mock import" in v for v in prod_violations)

  test_dir = tmp_path / "tests"
  test_dir.mkdir(parents=True)
  test_mock_file = test_dir / "test_mod.py"
  test_mock_file.write_text(
    nl.join(['"""Test module."""', "from unittest.mock import MagicMock", "m = MagicMock()"]),
    encoding="utf-8",
  )
  assert check_source(test_mock_file.read_text(encoding="utf-8"), test_mock_file) == []

  test_gs_file = test_dir / "test_gs.py"
  test_gs_code = nl.join(['"""Test getattr_static ban."""', "import inspect", "x = inspect.getattr_static"])
  test_gs_file.write_text(test_gs_code, encoding="utf-8")
  assert any("inspect.getattr_static" in v for v in check_source(test_gs_code, test_gs_file))


def test_lint_typing_dunder_main(tmp_path: Path) -> None:
  """Verify executing `scripts.lint_typing` as `__main__` exits cleanly on valid code.

  Args:
      tmp_path: Temporary directory path fixture.
  """
  ok_file = tmp_path / "ok.py"
  ok_file.write_text(chr(10).join(['"""Clean module."""', "val: int = 42", ""]), encoding="utf-8")
  with patch.object(sys, "argv", ["scripts/lint_typing.py", str(ok_file)]):
    with pytest.raises(SystemExit) as exc_info:
      runpy.run_module("scripts.lint_typing", run_name="__main__")
    assert exc_info.value.code == 0
