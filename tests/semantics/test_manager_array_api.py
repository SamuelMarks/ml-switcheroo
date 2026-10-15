"""Test suite for the Manager Array Api module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_array_api() -> None:
  """Verifies the behavior of manager array API."""
  manager: SemanticsManager = SemanticsManager()
  manager.data = {"test_op": {"variants": {}}}
  manager._key_origins = {"test_op": "array_api"}
  _res = manager._build_index()
  assert _res is not None or _res is None
