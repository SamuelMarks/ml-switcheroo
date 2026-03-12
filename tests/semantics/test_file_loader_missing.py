import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader


class DummyManager:
  def __init__(self):
    self.data = {}
    self._key_origins = {}
    self.framework_configs = {}
    self.test_templates = {}


def test_file_loader_discovered_filename(tmp_path):
  manager = DummyManager()
  loader = KnowledgeBaseLoader(manager)

  # Create semantics dir
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_discovered.json").write_text('{"test": {}}')

  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      loader.load_knowledge_graph()

  # Check that it processed the file
  assert "test" in manager.data


def test_file_loader_spec_exception(tmp_path, capsys):
  manager = DummyManager()
  loader = KnowledgeBaseLoader(manager)

  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_neural.json").write_text("invalid json")

  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      loader.load_knowledge_graph()

  captured = capsys.readouterr()
  assert "⚠️ Error loading" in captured.out


def test_file_loader_overlay_exception(tmp_path, capsys):
  manager = DummyManager()
  loader = KnowledgeBaseLoader(manager)

  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "test_map.json").write_text("invalid json")

  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=tmp_path / "semantics"):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      loader.load_knowledge_graph()

  captured = capsys.readouterr()
  assert "⚠️ Error loading overlay test_map.json" in captured.out
