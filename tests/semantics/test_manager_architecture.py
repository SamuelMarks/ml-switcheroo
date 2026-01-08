"""
Architecture Tests for SemanticsManager.
"""

from unittest.mock import patch

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_loads_files():
  """Verify manager attempts to load JSONs from semantics dir."""

  # Mock data
  mock_math = {"Abs": {"description": "Absolute value", "std_args": ["x"]}}

  # We patch the KnowledgeBaseLoader internal logic or the open calls
  # Simpler to patch KnowledgeBaseLoader._load_tier_content to see if it receives data

  with patch("ml_switcheroo.semantics.manager.KnowledgeBaseLoader.load_knowledge_graph") as mock_load:
    # Mock registry loader to avoid side effects
    with patch("ml_switcheroo.semantics.manager.RegistryLoader.hydrate"):
      mgr = SemanticsManager()

      # Should call loading sequence
      mock_load.assert_called_once()


def test_clean_slate_if_files_missing(tmp_path):
  """
  Scenario: Semantics directory is empty/missing.
  Expectation: Manager initializes empty without crashing.
  """
  empty_sem = tmp_path / "semantics"
  empty_sem.mkdir()

  # Patch path resolver to point to temp empty dir
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=empty_sem):
    # Also patch snapshot dir
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      with patch("ml_switcheroo.semantics.manager.RegistryLoader.hydrate"):
        mgr = SemanticsManager()

        # Should be empty
        assert mgr.data == {}
        assert mgr.get_known_apis() == {}
