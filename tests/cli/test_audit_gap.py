import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from ml_switcheroo.cli.handlers.audit import resolve_roots, handle_audit


def test_resolve_roots_no_adapter():
  with patch("ml_switcheroo.cli.handlers.audit.get_adapter", return_value=None):
    res = resolve_roots(["unknown"])
    assert res == {"unknown"}


def test_handle_audit_path_not_found():
  with patch("pathlib.Path.exists", return_value=False):
    res = handle_audit(Path("fake"), ["torch"], False)
    assert res == 1


def test_handle_audit_parse_error():
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("pathlib.Path.is_file", return_value=True),
    patch("ml_switcheroo.cli.handlers.audit.SemanticsManager"),
    patch("builtins.open", mock_open(read_data="def foo():\n")),
    patch("libcst.parse_module", side_effect=Exception("parse error")),
  ):
    # JSON mode to hit some lines and avoid stdout parsing
    # Actually want to hit line 96-98
    res = handle_audit(Path("fake.py"), ["torch"], False)
    assert res == 0  # no missing ops because none parsed


def test_handle_audit_missing_ops():
  import tempfile

  with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
    tf.write(b"def foo(): pass\n")
    tf.flush()
    real_path = Path(tf.name)
  try:
    with (
      patch("ml_switcheroo.cli.handlers.audit.SemanticsManager"),
      patch("libcst.parse_module"),
      patch("ml_switcheroo.cli.handlers.audit.CoverageScanner") as MockScanner,
    ):
      MockScanner.return_value.results = {"torch.missing": (False, "torch"), "torch.supported": (True, "torch")}
      res = handle_audit(real_path, ["torch"], False)
      assert res == 1
  finally:
    real_path.unlink()


def test_handle_audit_json_mode():
  import tempfile

  with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
    tf.write(b"def foo(): pass\n")
    tf.flush()
    real_path = Path(tf.name)
  try:
    with (
      patch("ml_switcheroo.cli.handlers.audit.SemanticsManager"),
      patch("libcst.parse_module"),
      patch("ml_switcheroo.cli.handlers.audit.CoverageScanner") as MockScanner,
    ):
      MockScanner.return_value.results = {"torch.missing": (False, "torch"), "torch.supported": (True, "torch")}
      # True for json_mode
      res = handle_audit(real_path, ["torch"], True)
      assert res == 1
  finally:
    real_path.unlink()
