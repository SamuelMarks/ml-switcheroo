"""Test suite for the Manager Overlay Loading module."""

import json
import pytest
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


@pytest.fixture
def mock_root_tree(tmp_path):
  """Provides a mock root tree for testing."""
  semantics_dir = tmp_path / "semantics"
  semantics_dir.mkdir()
  snapshots_dir = tmp_path / "snapshots"
  snapshots_dir.mkdir()
  spec_content = {
    "Abs": {"description": "Calculate absolute value", "std_args": ["x"]},
    "Add": {"description": "Addition", "std_args": ["a", "b"]},
  }
  import yaml

  odl_dir = semantics_dir / "odl"
  odl_dir.mkdir()
  (odl_dir / "add.yaml").write_text(yaml.dump({"Add": spec_content["Add"]}))
  (odl_dir / "abs.yaml").write_text(yaml.dump({"Abs": spec_content["Abs"]}))
  torch_map = {
    "__framework__": "torch",
    "mappings": {"Abs": {"api": "torch.abs"}, "Add": {"api": "torch.add"}, "custom_op": {"api": "torch.special"}},
  }
  (snapshots_dir / "torch_vlatest_map.json").write_text(json.dumps(torch_map))
  jax_map = {"__framework__": "jax", "mappings": {"Abs": {"api": "jax.numpy.abs"}, "Add": {"api": "jax.numpy.add"}}}
  (snapshots_dir / "jax_vlatest_map.json").write_text(json.dumps(jax_map))
  return semantics_dir


@pytest.fixture
def manager(mock_root_tree):
  """Provides a mock manager for testing."""
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=mock_root_tree):
    with patch(
      "ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=mock_root_tree.parent / "snapshots"
    ):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        yield SemanticsManager()


def test_overlay_merging_logic(manager):
  """Verifies the behavior of overlay merging logic."""
  assert "Abs" in manager.data
  entry = manager.data["Abs"]
  assert "torch" in entry["variants"]
  assert entry["variants"]["torch"]["api"] == "torch.abs"
  assert "jax" in entry["variants"]
  assert entry["variants"]["jax"]["api"] == "jax.numpy.abs"


def test_overlay_missing_op_handling(manager):
  """Verifies the behavior of overlay missing op handling."""
  assert "custom_op" in manager.data
  entry = manager.data["custom_op"]
  assert manager._key_origins["custom_op"] == SemanticTier.EXTRAS.value
  assert entry["variants"]["torch"]["api"] == "torch.special"
  assert "Auto-generated" in entry["description"]


def test_filename_framework_inference(tmp_path):
  """Verifies the behavior of filename framework inference."""
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()
  (sem_dir / "k_math.json").write_text(json.dumps({"Sin": {}}))
  numpy_map = {"mappings": {"Sin": {"api": "numpy.sin"}}}
  (snap_dir / "numpy_vlatest_map.json").write_text(json.dumps(numpy_map))
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        mgr = SemanticsManager()
        mgr._reverse_index = {}
        assert "Sin" in mgr.data
        assert "numpy" in mgr.data["Sin"]["variants"]
        assert mgr.data["Sin"]["variants"]["numpy"]["api"] == "numpy.sin"


def test_reverse_index_integrity(manager):
  """Verifies the behavior of reverse index integrity."""
  lookup = manager.get_definition("torch.abs")
  assert lookup is not None
  (abstract_id, data) = lookup
  assert abstract_id == "Abs"
  assert data["variants"]["torch"]["api"] == "torch.abs"
