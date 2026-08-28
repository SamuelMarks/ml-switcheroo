"""Test suite for the File Loader Missing module."""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader


class DummyManager:
  """Dummy Manager class for testing purposes."""

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
