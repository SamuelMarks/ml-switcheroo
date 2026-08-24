"""Test module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_semantic_manager_inherit_fallback() -> None:
  """Test element."""
  manager = SemanticsManager()
  res = manager._resolve_inheritance("unknown_fw")
  assert res is None


def test_semantic_manager_reverse_lookup_fallback() -> None:
  """Test element."""
  manager = SemanticsManager()
  manager.data = {"AbstractOp": {"description": "A test op"}}

  res = manager.get_definition("AbstractOp")
  assert res == ("AbstractOp", {"description": "A test op"})
