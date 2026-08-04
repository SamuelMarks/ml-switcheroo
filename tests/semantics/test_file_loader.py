"""Test suite for the File Loader module."""

import json
import yaml
from unittest.mock import MagicMock, patch
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_file_loader_init():
  """Verifies the behavior of file loader initialization."""
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  assert loader.mgr is mgr


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_missing_dirs(mock_snap_dir, mock_sem_dir, tmp_path):
  """Loads knowledge graph missing dirs."""
  mock_sem_dir.return_value = tmp_path / "missing_sem"
  mock_snap_dir.return_value = tmp_path / "missing_snap"
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  loader.load_knowledge_graph()


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_with_files(mock_snap_dir, mock_sem_dir, tmp_path):
  """Loads knowledge graph with files."""
  sem_dir = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir
  (sem_dir / "schema.yaml").touch()
  array_file = sem_dir / "array.yaml"
  array_content = {"Add": {"operation": "Add", "description": "add"}}
  array_file.write_text(yaml.dump(array_content))
  neural_file = sem_dir / "neural.yaml"
  neural_content = {"operation": "Conv2d", "description": "conv"}
  neural_file.write_text(yaml.dump(neural_content))
  other_file = sem_dir / "other.yaml"
  other_file.write_text(yaml.dump({"Other": {}}))
  disc_file = sem_dir / "k_discovered.yaml"
  disc_file.write_text(yaml.dump({"Disc": {}}))
  map_file = snap_dir / "test_map.json"
  map_file.write_text(json.dumps({"mappings": {}}))
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  with (
    patch("ml_switcheroo.semantics.file_loader.merge_tier_data") as mock_merge_tier,
    patch("ml_switcheroo.semantics.file_loader.merge_overlay_data") as mock_merge_overlay,
  ):
    loader.load_knowledge_graph()
    assert mock_merge_tier.call_count == 4
    assert mock_merge_overlay.call_count == 1


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_errors(mock_snap_dir, mock_sem_dir, tmp_path):
  """Loads knowledge graph errors."""
  sem_dir = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir
  array_file = sem_dir / "array.yaml"
  array_file.write_text("invalid: yaml: :")
  map_file = snap_dir / "test_map.json"
  map_file.write_text("invalid json")
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  loader.load_knowledge_graph()


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_json_fallback(mock_snap_dir, mock_sem_dir, tmp_path):
  """Tests loading json file from semantics directory."""
  sem_dir = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir

  # To reach the else branch for JSON parsing, we need to mock Path.rglob
  # to return a file with a .json suffix, because rglob("*.yaml") wouldn't find it natively.
  json_file = sem_dir / "test.json"
  json_file.write_text(json.dumps({"Add": {"operation": "Add", "description": "json file"}}))

  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)

  with patch("pathlib.Path.rglob", return_value=[json_file]):
    with patch.object(loader, "_load_tier_content") as mock_load_tier:
      loader.load_knowledge_graph()
      mock_load_tier.assert_called_once()


def test_load_tier_content():
  """Loads tier content."""
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  with patch("ml_switcheroo.semantics.file_loader.merge_tier_data") as mock_merge:
    loader._load_tier_content({"a": 1}, SemanticTier.ARRAY_API)
    mock_merge.assert_called_once()


def test_load_overlay_content():
  """Loads overlay content."""
  mgr = MagicMock()
  loader = KnowledgeBaseLoader(mgr)
  with patch("ml_switcheroo.semantics.file_loader.merge_overlay_data") as mock_merge:
    loader._load_overlay_content({"a": 1}, "test_map.json")
    mock_merge.assert_called_once()
