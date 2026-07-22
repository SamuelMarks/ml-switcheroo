"""Test suite for the Manager Architecture module."""

from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_loads_files():
  """Verifies the behavior of manager loads files."""
  with patch("ml_switcheroo.semantics.manager.KnowledgeBaseLoader.load_knowledge_graph") as mock_load:
    with patch("ml_switcheroo.semantics.manager.RegistryLoader.hydrate"):
      SemanticsManager()
      mock_load.assert_called_once()


def test_clean_slate_if_files_missing(tmp_path):
  """Verifies the behavior of clean slate if files missing."""
  empty_sem = tmp_path / "semantics"
  empty_sem.mkdir()
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=empty_sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      with patch("ml_switcheroo.semantics.manager.RegistryLoader.hydrate"):
        mgr = SemanticsManager()
        assert mgr.data == {}
        assert mgr.get_known_apis() == {}
