"""Test suite for the Batch Runner module."""

from unittest.mock import MagicMock
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.semantics.manager import SemanticsManager
from typing import List, Dict, Any, Tuple, Set


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
