"""Test module."""

from unittest.mock import MagicMock, patch
from ml_switcheroo.testing.batch_runner import BatchValidator
from pathlib import Path
from typing import Tuple, List, Dict, Any


def test_batch_runner_extract_sig() -> None:
  """Test element."""
  runner: BatchValidator = BatchValidator(MagicMock())
  runner.semantics.get_all_operations.return_value = ["foo"]  # type: ignore

  res: Tuple[List[str], Dict[str, str], Dict[str, Any]] = runner._unpack_args([{}])
  assert res == ([], {}, {})

  res2: Tuple[List[str], Dict[str, str], Dict[str, Any]] = runner._unpack_args(
    [
      {
        "name": "x",
        "type": "int",
        "min": 1,
        "max": 10,
        "default": 5,
        "options": [1, 2],
        "rank": 2,
        "dtype": "float32",
        "shape_spec": "N,C",
      }
    ]
  )
  assert res2[0] == ["x"]
  assert res2[1] == {"x": "int"}
  assert "x" in res2[2]
  assert res2[2]["x"]["min"] == 1


def test_batch_runner_scan_manual_tests() -> None:
  """Test element."""
  runner: BatchValidator = BatchValidator(MagicMock())
  runner.semantics.get_all_operations.return_value = ["foo"]  # type: ignore
  mock_file: MagicMock = MagicMock()
  mock_file.parts = ["test_foo.py"]
  mock_file.read_text.return_value = "def test_foo(): pass"
  with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.rglob", return_value=[mock_file]):
    res: set = runner._scan_manual_tests(Path("/root"))
    assert "foo" in res

    mock_file.read_text.side_effect = Exception("mock")
    runner._scan_manual_tests(Path("/root"))

    mock_file.read_text.side_effect = None
    mock_file.read_text.return_value = "def test_foo(:"
    runner._scan_manual_tests(Path("/root"))


def test_batch_runner_run() -> None:
  """Test element."""
  runner: BatchValidator = BatchValidator(MagicMock())
  runner.semantics.get_all_operations.return_value = ["foo"]  # type: ignore

  runner._run_manual_tests = MagicMock(return_value=(0, 0, []))  # type: ignore
  runner._scan_manual_tests = MagicMock(return_value={"foo"})  # type: ignore

  runner.run_all(manual_test_dir=Path("mock_dir"))

  with patch("ml_switcheroo.testing.batch_runner.track", return_value=["bar"]):
    runner._verify_operation = MagicMock()  # type: ignore
    runner.run_all(manual_test_dir=None, verbose=True)
