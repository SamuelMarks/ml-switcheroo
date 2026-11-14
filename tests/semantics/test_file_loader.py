"""Unit tests for `KnowledgeBaseLoader` using real `tmp_path` directories (`ml_switcheroo.semantics.file_loader`)."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

import ml_switcheroo.semantics.file_loader as fl_mod
from ml_switcheroo.semantics.file_loader import (
  KnowledgeBaseLoader,
  clean_content_descriptions,
  clean_sphinx_roles,
  get_docstring_parser,
  parse_docstring_sections,
)
from ml_switcheroo.semantics.manager import SemanticsManager


def test_load_knowledge_graph_missing_and_populated_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `KnowledgeBaseLoader.load_knowledge_graph` across missing, YAML, overlay, and discovered files.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  # 1. Missing semantics and snapshots directories -> clean no-op
  mgr_empty = SemanticsManager(load_defaults=False)
  monkeypatch.setattr(fl_mod, "resolve_semantics_dir", lambda: tmp_path / "missing_sem")
  monkeypatch.setattr(fl_mod, "resolve_snapshots_dir", lambda: tmp_path / "missing_snap")
  loader_empty = KnowledgeBaseLoader(mgr_empty)
  assert loader_empty.mgr is mgr_empty
  loader_empty.load_knowledge_graph()
  assert mgr_empty.data == {}

  # 2. Populated semantics (`array`, `neural`, `discovered`, `other`, skipped `schema.yaml`) & snapshot overlay
  sem_dir = tmp_path / "sem"
  snap_dir = tmp_path / "snap"
  sem_dir.mkdir()
  snap_dir.mkdir()
  monkeypatch.setattr(fl_mod, "resolve_semantics_dir", lambda: sem_dir)
  monkeypatch.setattr(fl_mod, "resolve_snapshots_dir", lambda: snap_dir)

  (sem_dir / "schema.yaml").write_text("ignored: true", encoding="utf-8")
  (sem_dir / "k_array_api.yaml").write_text(
    yaml.dump(
      {"Add": {"operation": "Add", "description": ":class:`~torch.Tensor` add", "std_args": ["x", "y"], "variants": {}}}
    ),
    encoding="utf-8",
  )
  (sem_dir / "Conv2d_neural.yaml").write_text(
    yaml.dump({"operation": "Conv2d", "description": ":func:`conv2d` layer", "std_args": ["x"], "variants": {}}),
    encoding="utf-8",
  )
  (sem_dir / "k_discovered.yaml").write_text(
    yaml.dump({"operation": "DiscOp", "description": "discovered", "std_args": [], "variants": {}}),
    encoding="utf-8",
  )
  (sem_dir / "k_extras.yaml").write_text(
    yaml.dump({"Save": {"operation": "Save", "description": "save op", "std_args": [], "variants": {}}}),
    encoding="utf-8",
  )
  (snap_dir / "torch_v2.0_map.json").write_text(
    json.dumps(
      {
        "__framework__": "torch",
        "mappings": {"Add": {"api": "torch.add", "description": ":func:`add`"}},
      }
    ),
    encoding="utf-8",
  )

  mgr = SemanticsManager(load_defaults=False)
  KnowledgeBaseLoader(mgr).load_knowledge_graph()
  assert {"Add", "Conv2d", "DiscOp", "Save"} <= set(mgr.data.keys())
  assert mgr.data["Add"]["description"] == "torch.Tensor add"
  assert mgr.data["Conv2d"]["description"] == "conv2d layer"


def test_load_knowledge_graph_json_fallback_and_malformed_files(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  """Verify `.json` spec fallback and logged warnings on malformed YAML/JSON files.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
      caplog: Pytest log capture fixture.
  """
  sem_dir = tmp_path / "sem_err"
  snap_dir = tmp_path / "snap_err"
  sem_dir.mkdir()
  snap_dir.mkdir()
  monkeypatch.setattr(fl_mod, "resolve_semantics_dir", lambda: sem_dir)
  monkeypatch.setattr(fl_mod, "resolve_snapshots_dir", lambda: snap_dir)

  (sem_dir / "broken.yaml").write_text("invalid: [yaml", encoding="utf-8")
  (snap_dir / "broken_map.json").write_text("{invalid json", encoding="utf-8")

  mgr = SemanticsManager(load_defaults=False)
  loader = KnowledgeBaseLoader(mgr)
  loader.load_knowledge_graph()
  assert "Error loading" in caplog.text
  assert "Error loading overlay broken_map.json" in caplog.text

  # `.json` fallback branch inside `_load_semantics_directory` when a `.json` path is yielded
  json_spec = sem_dir / "legacy_array.json"
  json_spec.write_text(
    json.dumps({"JsonOp": {"operation": "JsonOp", "description": "from json", "std_args": [], "variants": {}}}),
    encoding="utf-8",
  )
  orig_rglob = Path.rglob

  def _rglob_with_json(self_path: Path, pattern: str) -> list[Path]:
    """Return `[json_spec]` when scanning `sem_dir` for `*.yaml`.

    Args:
        self_path: Directory path being globbed.
        pattern: Glob pattern string.

    Returns:
        Matching paths list.
    """
    if self_path == sem_dir and pattern == "*.yaml":
      return [json_spec]
    return list(orig_rglob(self_path, pattern))

  monkeypatch.setattr(Path, "rglob", _rglob_with_json)
  loader.load_knowledge_graph()
  assert "JsonOp" in mgr.data


def test_sphinx_role_cleaning_and_docstring_helpers() -> None:
  """Verify `clean_sphinx_roles`, `clean_content_descriptions`, and Griffe docstring helpers."""
  assert clean_sphinx_roles(None) is None
  assert clean_sphinx_roles("plain string") == "plain string"
  assert clean_sphinx_roles(":class:`~torch.Tensor` and :func:`relu`") == "torch.Tensor and relu"

  _ = get_docstring_parser("google")
  sections = parse_docstring_sections(
    """Short description.

Args:
    x: Input tensor.
"""
  )
  assert isinstance(sections, list)

  clean_content_descriptions("not a dict")
  clean_content_descriptions([1, 2, 3])

  valid_entry: dict[str, JsonValue] = {"operation": "Baz", "description": ":func:`relu` op"}
  non_str_entry: dict[str, JsonValue] = {"operation": "Bar", "description": 123}
  data: dict[str, JsonValue] = {
    "raw_list": [1],
    "no_desc": {"operation": "Foo"},
    "non_str_desc": non_str_entry,
    "valid_desc": valid_entry,
  }
  clean_content_descriptions(data)
  assert valid_entry["description"] == "relu op"
  assert non_str_entry["description"] == 123

  # `SemanticLoadError` when `parse_docstring_with_griffe` returns a non-list
  from unittest.mock import MagicMock

  from ml_switcheroo.exceptions import SemanticLoadError

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(fl_mod, "parse_docstring_with_griffe", lambda _d, _p: "not_a_list")
    with pytest.raises(SemanticLoadError, match="Expected list"):
      parse_docstring_sections("doc")

  # Default `_DEFAULT_SEM_DIR` parsed YAML cache hit and untyped manager store fallbacks
  with pytest.MonkeyPatch.context() as mp:
    fake_default = Path("/fake/default/sem")
    mp.setattr(fl_mod, "_DEFAULT_SEM_DIR", fake_default)
    mp.setattr(
      fl_mod, "_PARSED_YAML_CACHE", [(10, {"CachedOp": {"operation": "CachedOp", "std_args": [], "variants": {}}})]
    )
    mp.setattr(fl_mod, "resolve_semantics_dir", lambda: fake_default)
    mp.setattr(fl_mod, "resolve_snapshots_dir", lambda: Path("/fake/missing/snaps"))
    mp.setattr(Path, "exists", lambda self_p: self_p == fake_default)

    cached_mgr = SemanticsManager(load_defaults=False)
    KnowledgeBaseLoader(cached_mgr).load_knowledge_graph()
    assert "CachedOp" in cached_mgr.data

    # Manager with non-dict `operations` and manager lacking `operations` attribute
    mock_non_dict_mgr = MagicMock(spec=SemanticsManager)
    mock_non_dict_mgr.data = {}
    mock_non_dict_mgr._key_origins = {}
    mock_non_dict_mgr.framework_configs = {}
    mock_non_dict_mgr.operations = MagicMock()
    mock_non_dict_mgr.framework_traits = {}
    KnowledgeBaseLoader(mock_non_dict_mgr).load_knowledge_graph()
    assert "CachedOp" in mock_non_dict_mgr.data

    bare_mgr = SemanticsManager(load_defaults=False)
    del bare_mgr.operations
    KnowledgeBaseLoader(bare_mgr).load_knowledge_graph()
    assert "CachedOp" in bare_mgr.data
