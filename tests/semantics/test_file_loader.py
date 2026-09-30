"""Test suite for the File Loader module."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.semantics.file_loader import (
  KnowledgeBaseLoader,
  clean_content_descriptions,
  clean_sphinx_roles,
  get_docstring_parser,
  parse_docstring_sections,
)


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
  content = {"Add": {"operation": "Add", "description": ":class:`~torch.Tensor` add"}}
  with patch("ml_switcheroo.semantics.file_loader.merge_tier_data") as mock_merge:
    loader._load_tier_content(content, SemanticTier.ARRAY_API)
    mock_merge.assert_called_once()
    assert content["Add"]["description"] == "torch.Tensor add"


def test_load_overlay_content() -> None:
  """Loads overlay content."""
  mgr: MagicMock = MagicMock()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(mgr)
  content = {"Add": {"operation": "Add", "description": ":func:`relu` activation"}}
  with patch("ml_switcheroo.semantics.file_loader.merge_overlay_data") as mock_merge:
    loader._load_overlay_content(content, "test_map.json")
    mock_merge.assert_called_once()
    assert content["Add"]["description"] == "relu activation"


def test_clean_sphinx_roles() -> None:
  """Verifies clean_sphinx_roles behavior on None and formatted strings."""
  assert clean_sphinx_roles(None) is None
  assert clean_sphinx_roles("plain string") == "plain string"
  res = clean_sphinx_roles(":class:`~torch.Tensor` and :func:`relu`")
  assert res == "torch.Tensor and relu"


def test_parse_docstring_sections_and_parser() -> None:
  """Verifies docstring parsing with Griffe integration."""
  parser = get_docstring_parser("google")
  assert parser is not None
  sections = parse_docstring_sections("Short description.\n\nArgs:\n    x: Input tensor.\n")
  assert isinstance(sections, list)


def test_clean_content_descriptions_branches() -> None:
  """Verifies clean_content_descriptions handles non-dicts and various description forms."""
  # Non-dict content
  clean_content_descriptions("not a dict")
  clean_content_descriptions([1, 2, 3])

  # Non-dict op, dict without description, dict with non-string desc, and valid desc
  data: dict = {
    "raw_list": [1],
    "no_desc": {"operation": "Foo"},
    "non_str_desc": {"operation": "Bar", "description": 123},
    "valid_desc": {"operation": "Baz", "description": ":func:`relu` op"},
  }
  clean_content_descriptions(data)
  assert data["valid_desc"]["description"] == "relu op"
  assert data["non_str_desc"]["description"] == 123


# --- Merged from test_file_loader_missing.py ---


class DummyManager:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummyManager instance."""
    self.data: Dict[str, Any] = {}
    self._key_origins: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}
    self.test_templates: Dict[str, Any] = {}


def test_file_loader_discovered_filename(tmp_path: Path) -> None:
  """Verifies the behavior of file loader discovered filename.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  manager: DummyManager = DummyManager()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(manager)  # type: ignore
  sem_dir: Path = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_discovered.yaml").write_text("operation: test\n")
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      loader.load_knowledge_graph()
  assert "test" in manager.data


def test_file_loader_spec_exception(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verifies the behavior of file loader spec correctly handling an exception.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  manager: DummyManager = DummyManager()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(manager)  # type: ignore
  sem_dir: Path = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_neural.yaml").write_text("invalid yaml:")
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=tmp_path / "snapshots"):
      loader.load_knowledge_graph()
  captured: pytest.CaptureResult[str] = capsys.readouterr()
  assert "⚠️ Error loading" in captured.out


def test_file_loader_overlay_exception(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verifies the behavior of file loader overlay correctly handling an exception.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  manager: DummyManager = DummyManager()
  loader: KnowledgeBaseLoader = KnowledgeBaseLoader(manager)  # type: ignore
  snap_dir: Path = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "test_map.json").write_text("invalid json")
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=tmp_path / "semantics"):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      loader.load_knowledge_graph()
  captured: pytest.CaptureResult[str] = capsys.readouterr()
  assert "⚠️ Error loading overlay test_map.json" in captured.out
