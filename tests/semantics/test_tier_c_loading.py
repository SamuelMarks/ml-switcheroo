import json
import pytest
from unittest.mock import patch, MagicMock
import libcst as cst
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.plugins.data_loader import transform_dataloader
from ml_switcheroo.core.hooks import _HOOKS


@pytest.fixture
def mock_specs(tmp_path):
  # Specs
  spec = {"CustomLoader": {"std_args": []}, "MagicContext": {"std_args": []}, "DataLoader": {"std_args": ["dataset"]}}
  (tmp_path / "semantics").mkdir()
  (tmp_path / "semantics" / "k_framework_extras.json").write_text(json.dumps(spec))

  # Overlays
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
      "CustomLoader": None,  # Explicit Disable
      "MagicContext": {"requires_plugin": "magic_shim"},
      "DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    },
  }
  (tmp_path / "snapshots" / "jax_vlatest_map.json").write_text(json.dumps(jax_map))

  return tmp_path


@pytest.fixture
def isolated_manager(mock_specs):
  sem = mock_specs / "semantics"
  snap = mock_specs / "snapshots"

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
      # Prevent default registry loading from polluting test
      with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
        return SemanticsManager()


def test_load_structure_from_extras(isolated_manager):
  # Check reverse lookup logic
  api = "torch.utils.data.DataLoader"

  # Since both CustomLoader and DataLoader map to same API, last one wins in reverse index?
  # Dictionary iteration order. DataLoader is later in file write above if sorted?
  # Or unordered. Let's check both possibilities or match based on 'api' uniqueness.
  # Let's rely on DataLoader being the "real" case we care about.

  defn = isolated_manager.get_definition(api)
  assert defn is not None
  abstract_id = defn[0]
  assert abstract_id in ["CustomLoader", "DataLoader"]


def test_rewriter_integration_null_variant(isolated_manager):
  # CustomLoader maps JAX to None -> Escape Hatch
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)

  # Force use of logic that resolves to CustomLoader?
  # Rewriter does: get_definition(api)
  # If we have ambiguity, we might get DataLoader which HAS a plugin.
  # We need to test the Null case specifically.

  # Let's forcefully point specific API to CustomLoader for this test
  # 'torch.custom.loader' -> 'CustomLoader' variant
  isolated_manager.data["CustomLoader"]["variants"]["torch"]["api"] = "torch.custom.loader"
  isolated_manager._reverse_index["torch.custom.loader"] = ("CustomLoader", isolated_manager.data["CustomLoader"])

  res = cst.parse_module("y = torch.custom.loader(x)").visit(rw).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res


def test_rewriter_integration_plugin_only(isolated_manager):
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)

  res = cst.parse_module("res = torch.magic()").visit(rw).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res
  assert "Missing required plugin" in res


def test_rewriter_integration_dataloader_shim(isolated_manager):
  # Must register hook for this test
  _HOOKS["convert_dataloader"] = transform_dataloader

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)

  # Ensure DataLoader reverse index matches correct standard
  isolated_manager._build_index()

  # NOTE: If multiple standards map to torch.utils.data.DataLoader, reverse index only keeps last one loaded.
  # In 'mock_specs' fixture, both CustomLoader and DataLoader map to it.
  # If CustomLoader wins (mapped to None), test fails. If DataLoader wins (Plugin), test passes.
  # We delete CustomLoader variant to ensure determinism.
  del isolated_manager.data["CustomLoader"]["variants"]["torch"]
  isolated_manager._build_index()

  code = "import torch\ndl = torch.utils.data.DataLoader(x)"
  res = cst.parse_module(code).visit(rw).code

  assert "class GenericDataLoader" in res
