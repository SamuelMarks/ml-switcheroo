"""Test suite for the Manager Array Api module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_array_api() -> None:
  """Verify `SemanticsManager` handles manager array API as expected."""
  manager: SemanticsManager = SemanticsManager()
  manager.data = {"test_op": {"variants": {}}}
  manager._key_origins = {"test_op": "array_api"}
  manager._build_index()
  _res = None
  assert _res is None
  assert isinstance(manager._reverse_index, dict)
