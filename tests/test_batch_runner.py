"""Test suite for the Batch Runner module."""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator


def test_unpack_args_mixed_support() -> None:
  """Verifies the behavior of unpack arguments mixed support."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  validator: BatchValidator = BatchValidator(mgr)
  raw_args: List[Any] = ["x", ("axis", "int")]
  (params, hints, constraints) = validator._unpack_args(raw_args)
  assert params == ["x", "axis"]
  assert hints == {"axis": "int"}
  assert "x" not in hints
  assert constraints == {}


def test_batch_runner_execution_flow() -> None:
  """Verifies the behavior of batch runner execution flow."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  mgr.get_known_apis.return_value = {
    "typed_op": {"variants": {}, "std_args": [("input", "Array"), ("dims", "Tuple[int]")]}
  }  # type: ignore
  validator: BatchValidator = BatchValidator(mgr)
  validator.runner.verify = MagicMock(return_value=(True, "OK"))  # type: ignore
  results: Dict[str, bool] = validator.run_all()
  assert results["typed_op"] is True
  validator.runner.verify.assert_called_once()  # type: ignore
  call_args: Tuple = validator.runner.verify.call_args  # type: ignore
  params_arg: List[str] = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["params"]
  hints_arg: Dict[str, str] = call_args[1].get("hints")
  constraints_arg: Dict[str, Any] = call_args[1].get("constraints")
  assert params_arg == ["input", "dims"]
  assert hints_arg == {"input": "Array", "dims": "Tuple[int]"}
  assert constraints_arg == {}


def test_skip_generated_tests(tmp_path: Any) -> None:
  """Verifies the behavior of skip generated tests.

  Args:
      tmp_path (Any): Tmp path pytest fixture.
  """
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  validator: BatchValidator = BatchValidator(mgr)
  valid_dir: Any = tmp_path / "valid"
  valid_dir.mkdir(parents=True, exist_ok=True)
  (valid_dir / "test_manual.py").write_text("def test_op(): pass", encoding="utf-8")
  gen_dir: Any = tmp_path / "generated"
  gen_dir.mkdir(parents=True, exist_ok=True)
  (gen_dir / "test_robotic.py").write_text("def test_skip_me(): pass", encoding="utf-8")
  found: Set[str] = validator._scan_manual_tests(tmp_path)
  assert "op" in found, "Failed to find manual test in 'valid' folder."
  assert "skip_me" not in found, "Incorrectly scanned a test from 'generated' folder."


# --- Merged from test_batch_runner_extra.py ---


def test_batch_runner_extract_sig() -> None:
  """Docstring."""
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
  """Docstring."""
  runner: BatchValidator = BatchValidator(MagicMock())
  runner.semantics.get_all_operations.return_value = ["foo"]  # type: ignore

  # Line 171: root does not exist
  with patch("pathlib.Path.exists", return_value=False):
    assert runner._scan_manual_tests(Path("/nonexistent")) == set()

  mock_file: MagicMock = MagicMock()
  mock_file.parts = ["test_foo.py"]
  mock_file.read_text.return_value = "x = 1\ndef helper(): pass\ndef test_gen_bar(): pass\ndef test_foo(): pass"
  with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.rglob", return_value=[mock_file]):
    res: set = runner._scan_manual_tests(Path("/root"))
    assert "foo" in res

    mock_file.read_text.side_effect = Exception("mock")
    runner._scan_manual_tests(Path("/root"))

    mock_file.read_text.side_effect = None
    mock_file.read_text.return_value = "def test_foo(:"
    runner._scan_manual_tests(Path("/root"))


def test_batch_runner_run() -> None:
  """Docstring."""
  mgr = MagicMock()
  mgr.get_known_apis.return_value = {"foo": {"variants": {}}}
  runner: BatchValidator = BatchValidator(mgr)

  runner._run_manual_tests = MagicMock(return_value=(0, 0, []))  # type: ignore
  runner._scan_manual_tests = MagicMock(return_value={"foo"})  # type: ignore

  # Lines 81-82: manual test priority matches op_name
  res = runner.run_all(manual_test_dir=Path("mock_dir"))
  assert res["foo"] is True

  with patch("ml_switcheroo.testing.batch_runner.track", return_value=["bar"]):
    mgr.get_known_apis.return_value = {"bar": {"variants": {}, "std_args": ["x"]}}
    runner.runner.verify = MagicMock(return_value=(True, "OK"))  # type: ignore
    runner.run_all(manual_test_dir=None, verbose=True)


def test_batch_runner_unpack_extra_branches() -> None:
  """Test unpack_args missing branches: 131->135, 141->123, 151->123."""
  runner: BatchValidator = BatchValidator(MagicMock())
  params, hints, constrs = runner._unpack_args([{"name": "x"}, 12345])
  assert params == ["x"]
  assert hints == {}
  assert constrs == {}
