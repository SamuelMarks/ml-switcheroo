"""Unit tests for `BatchValidator` (`ml_switcheroo.testing.batch_runner`)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator

_NL = chr(10)


def test_unpack_args_tuple_dict_and_scalar_variants() -> None:
  """Verify `BatchValidator._unpack_args` normalizes strings, tuples, constraint dicts, and invalid items."""
  validator = BatchValidator(SemanticsManager(load_defaults=False))

  params, hints, constraints = validator._unpack_args(["x", ("axis", "int"), {}, 12345])
  assert params == ["x", "axis"]
  assert hints == {"axis": "int"}
  assert constraints == {}

  rich_args: list[JsonValue] = [
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
    },
    {"name": "plain_dict"},
  ]
  p2, h2, c2 = validator._unpack_args(rich_args)
  assert p2 == ["x", "plain_dict"]
  assert h2 == {"x": "int"}
  assert c2["x"]["min"] == 1
  assert "plain_dict" not in c2


def test_scan_manual_tests_real_filesystem_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `_scan_manual_tests` skips `generated/`, malformed files, `OSError`, and missing roots.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  validator = BatchValidator(SemanticsManager(load_defaults=False))
  assert validator._scan_manual_tests(tmp_path / "nonexistent") == set()

  valid_dir = tmp_path / "valid"
  valid_dir.mkdir()
  (valid_dir / "test_manual.py").write_text(
    _NL.join(["x = 1", "def helper(): pass", "def test_gen_bar(): pass", "def test_op(): pass"]),
    encoding="utf-8",
  )
  (valid_dir / "test_bad_syntax.py").write_text(f"def test_broken(:{_NL}", encoding="utf-8")
  unreadable = valid_dir / "test_unreadable.py"
  unreadable.write_text("def test_secret(): pass", encoding="utf-8")

  gen_dir = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_robotic.py").write_text("def test_skip_me(): pass", encoding="utf-8")

  orig_read_text = Path.read_text

  def _guarded_read_text(self_path: Path, encoding: str | None = None, errors: str | None = None) -> str:
    """Raise `OSError` for `test_unreadable.py` while delegating all other reads.

    Args:
        self_path: File path being read.
        encoding: Text encoding.
        errors: Error handling mode.

    Returns:
        File contents string.

    Raises:
        OSError: When `self_path == unreadable`.
    """
    if self_path == unreadable:
      raise OSError("permission denied")
    return orig_read_text(self_path, encoding=encoding, errors=errors)

  monkeypatch.setattr(Path, "read_text", _guarded_read_text)
  found = validator._scan_manual_tests(tmp_path)
  assert found == {"op"}


def test_run_all_manual_priority_and_verbose_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `run_all` prioritizes manual tests, unpacks typed args into `runner.verify`, and supports `verbose`.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_known_apis.return_value = {
    "manual_op": {"variants": {}},
    "typed_op": {"variants": {}, "std_args": [("input", "Array"), ("dims", "Tuple[int]")]},
  }
  validator = BatchValidator(mgr)

  manual_dir = tmp_path / "manual"
  manual_dir.mkdir()
  (manual_dir / "test_manual_op.py").write_text("def test_manual_op(): pass", encoding="utf-8")

  mock_verify = MagicMock(return_value=(True, "OK"))
  monkeypatch.setattr(validator.runner, "verify", mock_verify)

  results = validator.run_all(manual_test_dir=manual_dir, verbose=False)
  assert results == {"manual_op": True, "typed_op": True}
  mock_verify.assert_called_once_with(
    {},
    ["input", "dims"],
    hints={"input": "Array", "dims": "Tuple[int]"},
    constraints={},
    shape_calc=None,
  )

  # Verbose mode with empty and non-empty catalogs
  mgr.get_known_apis.return_value = {}
  assert validator.run_all(verbose=True) == {}

  with patch("ml_switcheroo.testing.batch_runner.track", return_value=["bar"]):
    mgr.get_known_apis.return_value = {"bar": {"variants": {}, "std_args": ["x"]}}
    assert validator.run_all(manual_test_dir=None, verbose=True) == {"bar": True}
