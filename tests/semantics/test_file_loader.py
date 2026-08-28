"""Test suite for the File Loader module."""

import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_file_loader_init() -> None:
  """Verifies the behavior of file loader initialization."""
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  assert loader.mgr is mgr


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_missing_dirs(mock_snap_dir: MagicMock, mock_sem_dir: MagicMock, tmp_path: Path) -> None:
  """Loads knowledge graph missing dirs.

  Args:
      mock_snap_dir (MagicMock): Mock argument.
      mock_sem_dir (MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_sem_dir.return_value = tmp_path / "missing_sem"
  mock_snap_dir.return_value = tmp_path / "missing_snap"
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  loader.load_knowledge_graph()


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_with_files(mock_snap_dir: MagicMock, mock_sem_dir: MagicMock, tmp_path: Path) -> None:
  """Loads knowledge graph with files.

  Args:
      mock_snap_dir (MagicMock): Mock argument.
      mock_sem_dir (MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  sem_dir: Path = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir: Path = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir
  (sem_dir / "schema.yaml").touch()
  array_file: Path = sem_dir / "array.yaml"
  array_content: dict = {"Add": {"operation": "Add", "description": "add"}}
  array_file.write_text(yaml.dump(array_content))
  neural_file: Path = sem_dir / "neural.yaml"
  neural_content: dict = {"operation": "Conv2d", "description": "conv"}
  neural_file.write_text(yaml.dump(neural_content))
  other_file: Path = sem_dir / "other.yaml"
  other_file.write_text(yaml.dump({"Other": {}}))
  disc_file: Path = sem_dir / "k_discovered.yaml"
  disc_file.write_text(yaml.dump({"Disc": {}}))
  map_file: Path = snap_dir / "test_map.json"
  map_file.write_text(json.dumps({"mappings": {}}))
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  with (
    patch("ml_switcheroo.semantics.file_loader.merge_tier_data") as mock_merge_tier,
    patch("ml_switcheroo.semantics.file_loader.merge_overlay_data") as mock_merge_overlay,
  ):
    loader.load_knowledge_graph()
    assert mock_merge_tier.call_count == 4
    assert mock_merge_overlay.call_count == 1


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_errors(mock_snap_dir: MagicMock, mock_sem_dir: MagicMock, tmp_path: Path) -> None:
  """Loads knowledge graph errors.

  Args:
      mock_snap_dir (MagicMock): Mock argument.
      mock_sem_dir (MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  sem_dir: Path = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir: Path = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir
  array_file: Path = sem_dir / "array.yaml"
  array_file.write_text("invalid: yaml: :")
  map_file: Path = snap_dir / "test_map.json"
  map_file.write_text("invalid json")
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  loader.load_knowledge_graph()


@patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir")
@patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir")
def test_load_knowledge_graph_json_fallback(mock_snap_dir: MagicMock, mock_sem_dir: MagicMock, tmp_path: Path) -> None:
  """Tests loading json file from semantics directory.

  Args:
      mock_snap_dir (MagicMock): Mock argument.
      mock_sem_dir (MagicMock): Mock argument.
      tmp_path (Path): Tmp path pytest fixture.
  """
  sem_dir: Path = tmp_path / "sem"
  sem_dir.mkdir()
  mock_sem_dir.return_value = sem_dir
  snap_dir: Path = tmp_path / "snap"
  snap_dir.mkdir()
  mock_snap_dir.return_value = snap_dir

  # To reach the else branch for JSON parsing, we need to mock Path.rglob
  # to return a file with a .json suffix, because rglob("*.yaml") wouldn't find it natively.
  json_file: Path = sem_dir / "test.json"
  json_file.write_text(json.dumps({"Add": {"operation": "Add", "description": "json file"}}))

  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)

  with patch("pathlib.Path.rglob", return_value=[json_file]):
    with patch.object(loader, "_load_tier_content") as mock_load_tier:
      loader.load_knowledge_graph()
      mock_load_tier.assert_called_once()


def test_load_tier_content() -> None:
  """Loads tier content."""
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  with patch("ml_switcheroo.semantics.file_loader.merge_tier_data") as mock_merge:
    loader._load_tier_content({"a": 1}, SemanticTier.ARRAY_API)
    mock_merge.assert_called_once()


def test_load_overlay_content() -> None:
  """Loads overlay content."""
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  with patch("ml_switcheroo.semantics.file_loader.merge_overlay_data") as mock_merge:
    loader._load_overlay_content({"a": 1}, "test_map.json")
    mock_merge.assert_called_once()
