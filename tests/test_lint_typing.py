"""Tests for scripts/lint_typing.py."""

from __future__ import annotations

from pathlib import Path

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
      "from typing import Sequence",
      "from builtins import int",
      "from operator import add",
      "from os import path",
      "x: int = 1",
      "p = path.sep.upper()",
      "class S:",
      "  def __init__(self) -> None:",
      "    super().__setattr__('a', 1)",
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
      "from typing import Any, cast",
      "from builtins import getattr",
      "from operator import attrgetter",
      "import operator",
      "import typing",
      "import builtins",
      "x: Any = 1",
      "o: object = object()",
      "y: typing.Any = typing.cast(int, x)",
      "z = builtins.getattr(x, 'real')",
      "g = operator.attrgetter('real')(x)",
      "ig = attrgetter('real')(x)",
      "m = __import__('sys')",
      "c = compile('1', '<s >', 'eval')",
      "object.__setattr__(x, 'a', 1)",
      "type.__setattr__(int, 'a', 1)",
      "u = x.__add__(1)",
      "v = x.__call__()",
      "class C:",
      "  def __init__(self) -> None:",
      "    super().__init__()",
      "  def __getattr__(self, name: str) -> int:",
      "    return 0",
      "w = hasattr(x, 'real')  # type: ignore",
    ]
  )
  bad_file = tmp_path / "bad.py"
  bad_file.write_text(bad_code, encoding="utf-8")
  violations = check_source(bad_code, bad_file)
  assert len(violations) >= 18
  assert main([str(bad_file)]) == 1
  captured = capsys.readouterr()
  assert "Forbidden" in captured.err
