import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.discovery import handle_scaffold, handle_import_spec, _save_spec, handle_sync_standards
import json


def test_handle_scaffold():
  with (
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager"),
    patch("ml_switcheroo.cli.handlers.discovery.Scaffolder") as mock_scaffolder,
  ):
    res = handle_scaffold(["torch"], Path("/tmp/out"))
    assert res == 0
    mock_scaffolder.return_value.scaffold.assert_called_once()


def test_handle_import_spec_md_stablehlo(tmp_path):
  target = tmp_path / "spec.md"
  target.write_text("# StableHLO")

  with (
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.StableHloSpecImporter") as mock_importer,
    patch("ml_switcheroo.cli.handlers.discovery._save_spec") as mock_save,
  ):
    mock_resolve.return_value = tmp_path
    mock_importer.return_value.parse_file.return_value = {"a": 1}
    res = handle_import_spec(target)
    assert res == 0
    mock_save.assert_called_with(tmp_path, "k_stablehlo.json", {"a": 1})


def test_handle_import_spec_md_onnx(tmp_path):
  target = tmp_path / "spec.md"
  target.write_text("# ONNX Model")

  with (
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.OnnxSpecImporter") as mock_importer,
    patch("ml_switcheroo.cli.handlers.discovery._save_spec") as mock_save,
  ):
    mock_resolve.return_value = tmp_path
    mock_importer.return_value.parse_file.return_value = {"b": 2}
    res = handle_import_spec(target)
    assert res == 0
    mock_save.assert_called_with(tmp_path, "k_neural_net.json", {"b": 2})


def test_handle_import_spec_html_sass(tmp_path):
  target = tmp_path / "spec.html"
  target.write_text("<html>SASS</html>")

  with (
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.SassSpecImporter") as mock_importer,
    patch("ml_switcheroo.cli.handlers.discovery.get_definitions_path") as mock_get_def,
    patch("ml_switcheroo.cli.handlers.discovery._save_spec") as mock_save,
  ):
    mock_resolve.return_value = tmp_path
    mock_importer.return_value.parse_file.return_value = {"c": 3}
    mock_get_def.return_value = tmp_path / "sass.json"
    res = handle_import_spec(target)
    assert res == 0
    mock_save.assert_called_with(tmp_path, "sass.json", {"c": 3})


def test_handle_import_spec_dir_array_api(tmp_path):
  target = tmp_path / "stubs"
  target.mkdir()

  with (
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.ArrayApiSpecImporter") as mock_importer,
    patch("ml_switcheroo.cli.handlers.discovery._save_spec") as mock_save,
  ):
    mock_resolve.return_value = tmp_path
    mock_importer.return_value.parse_folder.return_value = {"d": 4}
    res = handle_import_spec(target)
    assert res == 0
    mock_save.assert_called_with(tmp_path, "k_array_api.json", {"d": 4})


def test_handle_import_spec_invalid(tmp_path):
  target = tmp_path / "spec.txt"
  target.write_text("invalid")
  res = handle_import_spec(target)
  assert res == 1


def test_handle_import_spec_md_read_fail(tmp_path):
  target = tmp_path / "spec.md"
  target.write_text("unreadable")

  with patch("pathlib.Path.read_text", side_effect=Exception("unreadable")):
    res = handle_import_spec(target)
    assert res == 1


def test_save_spec_new(tmp_path):
  out_dir = tmp_path / "out"
  _save_spec(out_dir, "test.json", {"a": 1})
  assert (out_dir / "test.json").exists()
  assert json.loads((out_dir / "test.json").read_text()) == {"a": 1}


def test_save_spec_merge(tmp_path):
  out_dir = tmp_path / "out"
  out_dir.mkdir()
  (out_dir / "test.json").write_text('{"a": 1}')
  _save_spec(out_dir, "test.json", {"b": 2})
  assert json.loads((out_dir / "test.json").read_text()) == {"a": 1, "b": 2}


def test_save_spec_merge_fail(tmp_path):
  out_dir = tmp_path / "out"
  out_dir.mkdir()
  (out_dir / "test.json").write_text("invalid json")
  _save_spec(out_dir, "test.json", {"b": 2})
  assert json.loads((out_dir / "test.json").read_text()) == {"b": 2}


def test_handle_sync_standards_no_valid_categories():
  res = handle_sync_standards(["invalid"], ["torch"], False)
  assert res == 1


def test_handle_sync_standards_dry_run(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.discovery.available_frameworks") as mock_fw,
    patch("ml_switcheroo.cli.handlers.discovery.get_adapter") as mock_get_adapter,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager") as mock_mgr,
    patch("ml_switcheroo.cli.handlers.discovery.ConsensusEngine") as mock_engine,
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
  ):
    mock_fw.return_value = ["torch", "jax"]

    mock_adapter_torch = MagicMock()
    mock_adapter_torch.collect_api.return_value = [MagicMock()]
    mock_adapter_jax = MagicMock()
    mock_adapter_jax.collect_api.return_value = [MagicMock()]

    def get_adapter_mock(fw):
      if fw == "torch":
        return mock_adapter_torch
      if fw == "jax":
        return mock_adapter_jax
      return None

    mock_get_adapter.side_effect = get_adapter_mock

    mock_mgr.return_value.get_known_apis.return_value = {"known_op": {}}

    mock_engine.return_value.cluster.return_value = []
    mock_cand = MagicMock()
    mock_cand.name = "new_op"
    mock_cand.std_args = {}
    mock_engine.return_value.filter_common.return_value = [mock_cand]

    mock_resolve.return_value = tmp_path

    res = handle_sync_standards(["layer"], None, True)
    assert res == 0


def test_handle_sync_standards_persist(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.discovery.available_frameworks") as mock_fw,
    patch("ml_switcheroo.cli.handlers.discovery.get_adapter") as mock_get_adapter,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager") as mock_mgr,
    patch("ml_switcheroo.cli.handlers.discovery.ConsensusEngine") as mock_engine,
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticPersister") as mock_persister,
  ):
    mock_fw.return_value = ["torch", "jax"]

    mock_adapter_torch = MagicMock()
    mock_adapter_torch.collect_api.return_value = [MagicMock()]
    mock_adapter_jax = MagicMock()
    mock_adapter_jax.collect_api.return_value = [MagicMock()]

    def get_adapter_mock(fw):
      if fw == "torch":
        return mock_adapter_torch
      if fw == "jax":
        return mock_adapter_jax
      return None

    mock_get_adapter.side_effect = get_adapter_mock

    mock_mgr.return_value.get_known_apis.return_value = {"known_op": {}}

    mock_cand1 = MagicMock()
    mock_cand1.name = "new_op"
    mock_cand1.std_args = {}
    mock_cand2 = MagicMock()
    mock_cand2.name = "known_op"  # Should be skipped

    mock_engine.return_value.filter_common.return_value = [mock_cand1, mock_cand2]

    mock_resolve.return_value = tmp_path

    res = handle_sync_standards(["layer"], ["torch", "jax"], False)
    assert res == 0
    mock_persister.return_value.persist.assert_called_once()


def test_handle_sync_standards_no_new_candidates(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.discovery.get_adapter") as mock_get_adapter,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager") as mock_mgr,
    patch("ml_switcheroo.cli.handlers.discovery.ConsensusEngine") as mock_engine,
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticPersister") as mock_persister,
  ):
    mock_adapter_torch = MagicMock()
    mock_adapter_torch.collect_api.return_value = [MagicMock()]
    mock_adapter_jax = MagicMock()
    mock_adapter_jax.collect_api.return_value = [MagicMock()]

    def get_adapter_mock(fw):
      if fw == "torch":
        return mock_adapter_torch
      if fw == "jax":
        return mock_adapter_jax
      return None

    mock_get_adapter.side_effect = get_adapter_mock

    mock_mgr.return_value.get_known_apis.return_value = {"known_op": {}}
    mock_cand1 = MagicMock()
    mock_cand1.name = "known_op"

    mock_engine.return_value.filter_common.return_value = [mock_cand1]

    mock_resolve.return_value = tmp_path

    res = handle_sync_standards(["layer"], ["torch", "jax"], False)
    assert res == 0
    mock_persister.return_value.persist.assert_not_called()


def test_handle_sync_standards_collect_fail(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.discovery.get_adapter") as mock_get_adapter,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager") as mock_mgr,
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
  ):
    mock_adapter_torch = MagicMock()
    mock_adapter_torch.collect_api.side_effect = Exception("fail")

    def get_adapter_mock(fw):
      if fw == "torch":
        return mock_adapter_torch
      return None

    mock_get_adapter.side_effect = get_adapter_mock

    mock_resolve.return_value = tmp_path

    res = handle_sync_standards(["layer"], ["torch"], False)
    assert res == 0


def test_handle_sync_standards_no_adapter(tmp_path):
  with (
    patch("ml_switcheroo.cli.handlers.discovery.get_adapter") as mock_get_adapter,
    patch("ml_switcheroo.cli.handlers.discovery.SemanticsManager") as mock_mgr,
    patch("ml_switcheroo.cli.handlers.discovery.resolve_semantics_dir") as mock_resolve,
  ):
    mock_get_adapter.return_value = None
    mock_resolve.return_value = tmp_path

    res = handle_sync_standards(["layer"], ["torch"], False)
    assert res == 0
