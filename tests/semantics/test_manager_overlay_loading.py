"""
Tests for Segmented Semantics Loading (Overlay Strategy).

Verifies that:
1. `SemanticsManager` loads base specs from `semantics/`.
2. `SemanticsManager` scans `snapshots/` for `*_mappings.json` files.
3. Variants in overlay files are merged into the main knowledge definition.
4. New operations in overlays (not in specs) are created as Extras.
5. Metadata (`__framework__`) in overlays is respected to assign variants.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


@pytest.fixture
def mock_root_tree(tmp_path):
  """
  Creates a mock directory structure mimicking the distributed source tree.

  /root
    /semantics
      k_array_api.json      <-- Spec Definitions
    /snapshots
      torch_mappings.json   <-- Framework Overlay
      jax_mappings.json     <-- Framework Overlay
  """
  semantics_dir = tmp_path / "semantics"
  semantics_dir.mkdir()
  snapshots_dir = tmp_path / "snapshots"
  snapshots_dir.mkdir()

  # 1. Create Base Spec (Spec-Only)
  # Note: No 'variants' defined here, just abstract contract.
  spec_content = {
    "Abs": {"description": "Calculate absolute value", "std_args": ["x"]},
    "Add": {"description": "Addition", "std_args": ["a", "b"]},
  }
  (semantics_dir / "k_array_api.json").write_text(json.dumps(spec_content))

  # 2. Create Torch Overlay
  # Uses dedicated __framework__ key
  torch_map = {
    "__framework__": "torch",
    "mappings": {
      "Abs": {"api": "torch.abs"},
      "Add": {"api": "torch.add"},
      # Op not in Spec
      "TorchOnlyOp": {"api": "torch.special"},
    },
  }
  (snapshots_dir / "torch_mappings.json").write_text(json.dumps(torch_map))

  # 3. Create JAX Overlay
  # Uses filename inference if __framework__ missing? No, we enforce explicit struct test first.
  jax_map = {"__framework__": "jax", "mappings": {"Abs": {"api": "jax.numpy.abs"}, "Add": {"api": "jax.numpy.add"}}}
  (snapshots_dir / "jax_mappings.json").write_text(json.dumps(jax_map))

  return semantics_dir


@pytest.fixture
def manager(mock_root_tree):
  """Initializes manager with patched path resolution."""

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=mock_root_tree):
    # Must patch available_frameworks to prevent loading defaults from real code registry
    with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
      yield SemanticsManager()


def test_overlay_merging_logic(manager):
  """
  Scenario: Load 'Abs' from Spec. Merge 'torch' and 'jax' from Snapshots.
  Expectation: 'Abs' entry contains both variants.
  """
  # 1. Verify Spec loaded
  assert "Abs" in manager.data
  entry = manager.data["Abs"]

  # 2. Verify Torch injected
  assert "torch" in entry["variants"]
  assert entry["variants"]["torch"]["api"] == "torch.abs"

  # 3. Verify JAX injected
  assert "jax" in entry["variants"]
  assert entry["variants"]["jax"]["api"] == "jax.numpy.abs"


def test_overlay_missing_op_handling(manager):
  """
  Scenario: Overlay defines 'TorchOnlyOp' which is NOT in the Spec files.
  Expectation: Manager creates a new entry (Tier: Extras).
  """
  assert "TorchOnlyOp" in manager.data
  entry = manager.data["TorchOnlyOp"]

  # Check source tracking
  assert manager._key_origins["TorchOnlyOp"] == SemanticTier.EXTRAS.value

  # Check variant
  assert entry["variants"]["torch"]["api"] == "torch.special"

  # Check generated description
  assert "Auto-generated" in entry["description"]


def test_filename_framework_inference(tmp_path):
  """
  Scenario: Overlay file lacks "__framework__" key.
  Expectation: Logic infers framework from filename 'numpy_mappings.json' -> 'numpy'.
  """
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()

  # Spec
  (sem_dir / "k_math.json").write_text(json.dumps({"Sin": {}}))

  # Overlay (No __framework__ key)
  numpy_map = {"mappings": {"Sin": {"api": "numpy.sin"}}}
  (snap_dir / "numpy_mappings.json").write_text(json.dumps(numpy_map))

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
      mgr = SemanticsManager()
      mgr._reverse_index = {}

      assert "Sin" in mgr.data
      assert "numpy" in mgr.data["Sin"]["variants"]
      assert mgr.data["Sin"]["variants"]["numpy"]["api"] == "numpy.sin"


def test_reverse_index_integrity(manager):
  """
  Scenario: Reverse check APIs loaded from Overlays.
  Expectation: `get_definition("torch.abs")` returns ("Abs", data).
  """
  # Check reverse lookup
  lookup = manager.get_definition("torch.abs")
  assert lookup is not None

  abstract_id, data = lookup
  assert abstract_id == "Abs"
  assert data["variants"]["torch"]["api"] == "torch.abs"
