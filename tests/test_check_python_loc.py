"""Unit and integration tests for scripts/check_python_loc.sh."""

from pathlib import Path
import subprocess


def test_check_python_loc_compliant(tmp_path: Path) -> None:
  """Test check_python_loc.sh exits with 0 on files within the line count limit.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "check_python_loc.sh"
  valid_py = tmp_path / "valid.py"
  valid_py.write_text("# short file" + chr(10) + "x = 1" + chr(10), encoding="utf-8")

  res = subprocess.run(["bash", str(script), str(valid_py)], capture_output=True, text=True)
  assert res.returncode == 0
  assert "Error:" not in res.stdout


def test_check_python_loc_non_python_files(tmp_path: Path) -> None:
  """Test check_python_loc.sh ignores non-Python files even if they exceed line limits.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "check_python_loc.sh"
  huge_txt = tmp_path / "huge.txt"
  huge_txt.write_text(("line" + chr(10)) * 2000, encoding="utf-8")

  res = subprocess.run(["bash", str(script), str(huge_txt)], capture_output=True, text=True)
  assert res.returncode == 0
  assert "Error:" not in res.stdout


def test_check_python_loc_overflow(tmp_path: Path) -> None:
  """Test check_python_loc.sh fails when a Python file exceeds 1500 lines.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "check_python_loc.sh"
  overflow_py = tmp_path / "overflow.py"
  overflow_py.write_text(("print('line')" + chr(10)) * 1501, encoding="utf-8")

  res = subprocess.run(["bash", str(script), str(overflow_py)], capture_output=True, text=True)
  assert res.returncode == 1
  assert "exceeds 1500 lines of code (1501 lines)" in res.stdout


def test_check_python_loc_multiple_files(tmp_path: Path) -> None:
  """Test check_python_loc.sh with multiple files including compliant and overflowing.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "check_python_loc.sh"
  valid_py = tmp_path / "valid.py"
  valid_py.write_text("a = 1" + chr(10), encoding="utf-8")
  overflow_py = tmp_path / "overflow.py"
  overflow_py.write_text(("b = 2" + chr(10)) * 1600, encoding="utf-8")

  res = subprocess.run(
    ["bash", str(script), str(valid_py), str(overflow_py)],
    capture_output=True,
    text=True,
  )
  assert res.returncode == 1
  assert "overflow.py exceeds 1500 lines" in res.stdout


def test_check_python_loc_empty_arguments() -> None:
  """Test check_python_loc.sh with zero arguments exits successfully."""
  script = Path(__file__).resolve().parent.parent / "scripts" / "check_python_loc.sh"
  res = subprocess.run(["bash", str(script)], capture_output=True, text=True)
  assert res.returncode == 0
