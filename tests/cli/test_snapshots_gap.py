import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.snapshots import (
  handle_snapshot,
  handle_sync,
  _get_pkg_version,
  _capture_framework,
  _save_snapshot,
)


def test_handle_snapshot_no_frameworks():
  with patch("ml_switcheroo.cli.handlers.snapshots.available_frameworks", return_value=[]):
    assert handle_snapshot(None) == 1


def test_handle_snapshot_no_data(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.snapshots.available_frameworks", return_value=["torch"]),
    patch("ml_switcheroo.cli.handlers.snapshots._capture_framework", return_value={}),
  ):
    assert handle_snapshot(tmp_path) == 1


def test_handle_snapshot_success(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.snapshots.available_frameworks", return_value=["torch"]),
    patch("ml_switcheroo.cli.handlers.snapshots._capture_framework", return_value={"version": "1.0"}),
    patch("ml_switcheroo.cli.handlers.snapshots._save_snapshot") as mock_save,
  ):
    assert handle_snapshot(tmp_path) == 0
    mock_save.assert_called_once()


def test_get_pkg_version():
  with patch("importlib.metadata.version", return_value="2.0"):
    assert _get_pkg_version("torch") == "2.0"
    assert _get_pkg_version("flax_nnx") == "2.0"


def test_get_pkg_version_error():
  with patch("importlib.metadata.version", side_effect=Exception):
    assert _get_pkg_version("unknown") == "unknown"


def test_capture_framework_no_adapter():
  with patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None):
    assert _capture_framework("unknown") == {}


def test_capture_framework_unknown_version():
  with (
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=MagicMock()),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="unknown"),
  ):
    assert _capture_framework("torch") == {}


def test_capture_framework_success():
  adapter = MagicMock()
  ref = MagicMock()
  ref.model_dump.return_value = {"a": 1}
  adapter.collect_api.return_value = [ref]

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=adapter),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
  ):
    res = _capture_framework("torch")
    assert res["version"] == "2.0"
    assert "categories" in res


def test_capture_framework_collect_fail():
  adapter = MagicMock()
  adapter.collect_api.side_effect = Exception("fail")

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=adapter),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
  ):
    res = _capture_framework("torch")
    assert res == {}


def test_save_snapshot_empty():
  _save_snapshot("torch", {}, Path("/tmp"))  # Should do nothing


def test_save_snapshot_success(tmp_path):
  _save_snapshot("torch", {"version": "2.0+cpu"}, tmp_path)
  assert (tmp_path / "torch_v2.0_cpu.json").exists()


def test_handle_sync_no_snap_path(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_array_api.json").write_text('{"op1": {}}')
  snap_dir = tmp_path / "snapshots"

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None),
    patch("ml_switcheroo.cli.handlers.snapshots.FrameworkSyncer") as mock_syncer,
  ):

    def sync_mock(data, fw):
      data["op1"]["variants"] = {fw: {"mapped": True}}

    mock_syncer.return_value.sync.side_effect = sync_mock

    assert handle_sync("torch") == 0
    assert (snap_dir / "torch_v2.0_map.json").exists()


def test_handle_sync_existing_snap_path(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "torch_v2.0_map.json").write_text('{"__framework__": "torch", "mappings": {"op2": {}}}')

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None),
  ):
    assert handle_sync("torch") == 0


def test_handle_sync_bad_existing_snap_path(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "torch_v2.0_map.json").write_text("invalid json")

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None),
  ):
    assert handle_sync("torch") == 0


def test_handle_sync_with_adapter_definitions(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"

  adapter = MagicMock()
  model = MagicMock()
  model.model_dump.return_value = {"static": True}
  adapter.definitions = {"op_static": model}
  adapter.test_config = {"a": 1}
  adapter.apply_wiring.return_value = None

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=adapter),
  ):
    assert handle_sync("torch") == 0
    import json

    data = json.loads((snap_dir / "torch_v2.0_map.json").read_text())
    assert data["mappings"]["op_static"] == {"static": True}
    assert data["templates"] == {"a": 1}
    adapter.apply_wiring.assert_called_once()


def test_handle_sync_spec_file_error(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  # Directory instead of file to trigger open() error
  (sem_dir / "k_array_api.json").mkdir()
  snap_dir = tmp_path / "snapshots"

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None),
  ):
    assert handle_sync("torch") == 0


def test_handle_sync_adapter_wiring_fails(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"

  adapter = MagicMock()
  adapter.definitions = {}
  adapter.test_config = {}
  adapter.apply_wiring.side_effect = Exception("wiring fail")

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=adapter),
  ):
    assert handle_sync("torch") == 0


def test_handle_sync_existing_mappings_in_tier_data(tmp_path):
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  (sem_dir / "k_array_api.json").write_text('{"op1": {}}')
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (snap_dir / "torch_v2.0_map.json").write_text('{"__framework__": "torch", "mappings": {"op1": {"existing": true}}}')

  with (
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.cli.handlers.snapshots.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.cli.handlers.snapshots._get_pkg_version", return_value="2.0"),
    patch("ml_switcheroo.cli.handlers.snapshots.get_adapter", return_value=None),
    patch("ml_switcheroo.cli.handlers.snapshots.FrameworkSyncer") as mock_syncer,
  ):
    assert handle_sync("torch") == 0

    args, _ = mock_syncer.return_value.sync.call_args
    tier_data = args[0]
    assert "variants" in tier_data["op1"]
    assert tier_data["op1"]["variants"]["torch"] == {"existing": True}
