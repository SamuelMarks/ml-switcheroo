"""Test suite for the Manager Array Api module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_array_api():
  """Verifies the behavior of manager array API."""
  manager = SemanticsManager()
  manager.data = {"test_op": {"variants": {}}}
  manager._key_origins = {"test_op": "array_api"}
  manager._build_index()
