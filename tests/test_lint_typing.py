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
  clean_file.write_text(f'"""Clean module."""{nl}{nl}x: int = 1{nl}', encoding="utf-8")
  assert check_source(clean_file.read_text(encoding="utf-8"), clean_file) == []
  assert collect_python_files([clean_file, tmp_path]) == [clean_file]
  assert main([str(clean_file)]) == 0

  bad_code = nl.join(
    [
      '"""Bad module."""',
      "from typing import Any, cast",
      "from builtins import getattr",
      "import typing",
      "import builtins",
      "x: Any = 1",
      "y: typing.Any = typing.cast(int, x)",
      "z = builtins.getattr(x, 'real')",
      "object.__setattr__(x, 'a', 1)",
      "type.__setattr__(int, 'a', 1)",
      "u = x.__add__(1)",
      "v = x.__call__()",
      "class C:",
      "  def __init__(self) -> None:",
      "    super().__init__()",
      "w = hasattr(x, 'real')  # type: ignore",
    ]
  )
  bad_file = tmp_path / "bad.py"
  bad_file.write_text(bad_code, encoding="utf-8")
  violations = check_source(bad_code, bad_file)
  assert len(violations) >= 11
  assert main([str(bad_file)]) == 1
  captured = capsys.readouterr()
  assert "Forbidden" in captured.err
