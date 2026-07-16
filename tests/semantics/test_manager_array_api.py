"""Test module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_array_api():
  """Test function."""
  manager = SemanticsManager()
  manager.data = {"test_op": {"variants": {}}}
  manager._key_origins = {"test_op": "array_api"}

  # Sorting logic in manager._build_index or similar triggers get_score
  manager._build_index()
