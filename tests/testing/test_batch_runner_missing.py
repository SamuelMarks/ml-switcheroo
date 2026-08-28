"""Test suite for the Batch Runner Missing module."""

import pathlib
from typing import Dict, Any, List


def test_batch_runner_verbose() -> None:
  """Verifies the behavior of batch runner verbose."""
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm: SemanticsManager = SemanticsManager()
  with __import__("unittest.mock").mock.patch.object(sm, "get_known_apis", return_value={}):
    validator: BatchValidator = BatchValidator(sm)
    validator.run_all(verbose=True)


def test_batch_runner_unpack_args_dict() -> None:
  """Verifies the behavior of batch runner unpack arguments dictionary."""
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator: BatchValidator = BatchValidator(SemanticsManager())
  raw_args: List[Any] = [{}, {"name": "a", "type": "int", "min": 0, "max": 10}, {"name": "b"}]
  p: List[str]
  h: Dict[str, str]
  c: Dict[str, Dict[str, Any]]
  p, h, c = validator._unpack_args(raw_args)
  assert "a" in p
  assert "b" in p
  assert h["a"] == "int"
  assert "min" in c["a"]


def test_batch_runner_scan_manual_tests_not_exist(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of batch runner scan manual tests not exist."""
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator: BatchValidator = BatchValidator(SemanticsManager())
  assert validator._scan_manual_tests(tmp_path / "fake_dir") == set()


def test_batch_runner_scan_manual_tests_parse_error(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of batch runner scan manual tests parse correctly handling an error."""
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator: BatchValidator = BatchValidator(SemanticsManager())
  d: pathlib.Path = tmp_path / "test_dir"
  d.mkdir()
  f: pathlib.Path = d / "test_bad.py"
  f.write_text("def test_foo():\n    this is a syntax error\n")
  assert validator._scan_manual_tests(d) == set()
