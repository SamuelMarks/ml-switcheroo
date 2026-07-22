"""Test suite for the Tier C Loading module."""

import json
import pytest
from unittest.mock import patch
import libcst as cst
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.data_loader import transform_dataloader
from ml_switcheroo.core.hooks import _HOOKS


@pytest.fixture
def mock_specs(tmp_path):
  """Provides a mock specs for testing."""
  spec = {"CustomLoader": {"std_args": []}, "MagicContext": {"std_args": []}, "DataLoader": {"std_args": ["dataset"]}}
  (tmp_path / "semantics").mkdir()
  import yaml

  odl_dir = tmp_path / "semantics" / "odl"
  odl_dir.mkdir()
  for k, v in spec.items():
    v["operation"] = k
    (odl_dir / f"{k}.yaml").write_text(yaml.dump(v))
  (tmp_path / "snapshots").mkdir()
  torch_map = {
    "__framework__": "torch",
    "mappings": {
      "CustomLoader": {"api": "torch.utils.data.DataLoader"},
      "DataLoader": {"api": "torch.utils.data.DataLoader"},
      "MagicContext": {"api": "torch.magic"},
    },
  }
  (tmp_path / "snapshots" / "torch_vlatest_map.json").write_text(json.dumps(torch_map))
  jax_map = {
    "__framework__": "jax",
    "mappings": {
      "CustomLoader": None,
      "MagicContext": {"requires_plugin": "magic_shim"},
      "DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    },
  }
  (tmp_path / "snapshots" / "jax_vlatest_map.json").write_text(json.dumps(jax_map))
  return tmp_path


@pytest.fixture
def isolated_manager(mock_specs):
  """Provides a mock isolated manager for testing."""
  sem = mock_specs / "semantics"
  snap = mock_specs / "snapshots"
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        return SemanticsManager()


def test_load_structure_from_extras(isolated_manager):
  """Loads structure from extras."""
  api = "torch.utils.data.DataLoader"
  defn = isolated_manager.get_definition(api)
  assert defn is not None
  abstract_id = defn[0]
  assert abstract_id in ["CustomLoader", "DataLoader"]


def test_rewriter_integration_null_variant(isolated_manager):
  """Verifies the behavior of rewriter integration null variant."""
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)
  isolated_manager.data["CustomLoader"]["variants"]["torch"]["api"] = "torch.custom.loader"
  isolated_manager._reverse_index["torch.custom.loader"] = ("CustomLoader", isolated_manager.data["CustomLoader"])
  res = rw.convert(cst.parse_module("y = torch.custom.loader(x)")).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res


def test_rewriter_integration_plugin_only(isolated_manager):
  """Verifies the behavior of rewriter integration plugin only."""
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)
  res = rw.convert(cst.parse_module("res = torch.magic()")).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res
  assert "Missing required plugin" in res


def test_rewriter_integration_dataloader_shim(isolated_manager):
  """Verifies the behavior of rewriter integration dataloader shim."""
  _HOOKS["convert_dataloader"] = transform_dataloader
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)
  isolated_manager._build_index()
  del isolated_manager.data["CustomLoader"]["variants"]["torch"]
  isolated_manager._build_index()
  code = "import torch\ndl = torch.utils.data.DataLoader(x)"
  res = rw.convert(cst.parse_module(code)).code
  assert "class GenericDataLoader" in res
