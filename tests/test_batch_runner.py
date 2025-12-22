"""
Tests for BatchValidator Argument Parsing and Discovery.

Verifies:
1. Legacy support: `["x", "y"]` -> params=["x", "y"], hints={}.
2. Feature 027 support: `[("x", "int")]` -> params=["x"], hints={"x": "int"}.
3. Integration: Unpacked data passed correctly to EquivalenceRunner.
4. Robust scanning: Ignores 'generated' folders but finds valid manual tests.
"""

from unittest.mock import MagicMock

from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.semantics.manager import SemanticsManager


def test_unpack_args_mixed_support():
  """
  Verify parsing of mixed arg formats (tuple vs string).
  """
  mgr = MagicMock(spec=SemanticsManager)
  validator = BatchValidator(mgr)

  # Input: Mix of legacy string and new typed tuple
  raw_args = ["x", ("axis", "int")]

  params, hints, constraints = validator._unpack_args(raw_args)

  assert params == ["x", "axis"]
  assert hints == {"axis": "int"}
  assert "x" not in hints
  assert constraints == {}


def test_batch_runner_execution_flow():
  """
  Verify the Validator calls runner.verify() with unpacked hints.
  """
  # 1. Setup Mock Semantics Data (Typed)
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_known_apis.return_value = {
    "typed_op": {"variants": {}, "std_args": [("input", "Array"), ("dims", "Tuple[int]")]}
  }

  # 2. Setup Validator
  validator = BatchValidator(mgr)

  # 3. Mock the internal Runner
  validator.runner.verify = MagicMock(return_value=(True, "OK"))

  # 4. Execute
  results = validator.run_all()

  # 5. Assertions
  assert results["typed_op"] is True

  validator.runner.verify.assert_called_once()
  call_args = validator.runner.verify.call_args

  # Inspect arguments passed to runner.verify(variants, params, hints=...)
  params_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["params"]
  hints_arg = call_args[1].get("hints")
  constraints_arg = call_args[1].get("constraints")

  assert params_arg == ["input", "dims"]
  assert hints_arg == {"input": "Array", "dims": "Tuple[int]"}
  assert constraints_arg == {}


def test_skip_generated_tests(tmp_path):
  """
  Verify manual test scanning explicitly skips 'generated' folder contents.
  Ensures robust path checking avoids false positives on path substrings.
  """
  mgr = MagicMock(spec=SemanticsManager)
  validator = BatchValidator(mgr)

  # Structure:
  # root/
  #   valid/
  #     test_manual.py (Should trigger 'op')
  #   generated/
  #     test_robotic.py (Should SKIP 'skip_me')

  # Create valid dir
  valid_dir = tmp_path / "valid"
  valid_dir.mkdir(parents=True, exist_ok=True)
  (valid_dir / "test_manual.py").write_text("def test_op(): pass", encoding="utf-8")

  # Create generated dir
  gen_dir = tmp_path / "generated"
  gen_dir.mkdir(parents=True, exist_ok=True)
  (gen_dir / "test_robotic.py").write_text("def test_skip_me(): pass", encoding="utf-8")

  # Run scan
  found = validator._scan_manual_tests(tmp_path)

  # Assertions
  assert "op" in found, "Failed to find manual test in 'valid' folder."
  assert "skip_me" not in found, "Incorrectly scanned a test from 'generated' folder."
