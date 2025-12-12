"""
Tests for Semantic Persistence (Autogen) with Hub-and-Spoke Split Architecture.

Verifies:
1.  **Split Writes**: Abstract Definitions go to `semantics/`, Mappings go to `snapshots/`.
2.  **File Creation**: New snapshot files are created if missing.
3.  **Conflict Resolution**: Existing keys (Manual Overrides) are protected in both Hub and Spokes.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.discovery.consensus import CandidateStandard
from ml_switcheroo.core.ghost import GhostRef


@pytest.fixture
def persister():
  return SemanticPersister()


@pytest.fixture
def sample_candidate():
  """Returns a 'Huber' candidate with Torch and JAX variants."""
  c = CandidateStandard(
    name="Huber",
    std_args=["reduction"],
    variants={
      "torch": GhostRef(name="HuberLoss", api_path="torch.nn.HuberLoss", kind="class"),
      "jax": GhostRef(name="huber_loss", api_path="optax.huber_loss", kind="function"),
    },
    arg_mappings={"torch": {"reduction": "reduction"}, "jax": {"reduction": "reduction_mode"}},
    score=1.0,
  )
  return c


@pytest.fixture
def mock_fs(tmp_path):
  """Creates a mock filesystem structure."""
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir(parents=True)
  snap_dir.mkdir(parents=True)
  return sem_dir, snap_dir


def test_persist_hub_creation(persister, sample_candidate, mock_fs):
  """
  Scenario: Target spec file does not exist.
  Expectation:
      1. Spec file created in semantics/
      2. Contains "std_args" and "description".
      3. Does NOT contain "variants".
  """
  sem_dir, snap_dir = mock_fs
  target_spec = sem_dir / "k_generated.json"

  with patch("ml_switcheroo.semantics.autogen.resolve_snapshots_dir", return_value=snap_dir):
    persister.persist([sample_candidate], target_spec)

  assert target_spec.exists()
  data = json.loads(target_spec.read_text("utf-8"))

  assert "Huber" in data
  entry = data["Huber"]

  # Hub Requirements
  assert entry["std_args"] == ["reduction"]
  assert "Auto-discovered" in entry["description"]

  # Split Requirement: Variants should NOT be here
  assert "variants" not in entry


def test_persist_spoke_creation(persister, sample_candidate, mock_fs):
  """
  Scenario: Snapshots directory is empty.
  Expectation:
      1. `torch_vlatest_map.json` created.
      2. `jax_vlatest_map.json` created.
      3. Mappings contain `api` and `args`.
  """
  sem_dir, snap_dir = mock_fs
  target_spec = sem_dir / "k_generated.json"

  with patch("ml_switcheroo.semantics.autogen.resolve_snapshots_dir", return_value=snap_dir):
    persister.persist([sample_candidate], target_spec)

  # Check Torch Spoke
  torch_snap = snap_dir / "torch_vlatest_map.json"
  assert torch_snap.exists()

  t_data = json.loads(torch_snap.read_text())
  assert t_data["__framework__"] == "torch"
  assert "Huber" in t_data["mappings"]
  assert t_data["mappings"]["Huber"]["api"] == "torch.nn.HuberLoss"
  assert t_data["mappings"]["Huber"]["args"]["reduction"] == "reduction"

  # Check JAX Spoke
  jax_snap = snap_dir / "jax_vlatest_map.json"
  assert jax_snap.exists()
  j_data = json.loads(jax_snap.read_text())
  assert "Huber" in j_data["mappings"]
  assert j_data["mappings"]["Huber"]["args"]["reduction"] == "reduction_mode"


def test_manual_override_protection(persister, sample_candidate, mock_fs):
  """
  Scenario: 'Huber' exists in Spec (Hub) and Torch Snapshot (Spoke).
  Expectation:
      1. Hub entry is NOT updated.
      2. Torch Spoke entry is NOT updated.
      3. JAX Spoke entry IS created (it was missing).
  """
  sem_dir, snap_dir = mock_fs
  target_spec = sem_dir / "k_mixed.json"

  # Pre-seed Hub with manual edit
  hub_data = {"Huber": {"std_args": ["manual"], "description": "Manual Edit"}}
  target_spec.write_text(json.dumps(hub_data))

  # Pre-seed Torch Spoke with manual edit
  torch_snap = snap_dir / "torch_vlatest_map.json"
  torch_data = {"__framework__": "torch", "mappings": {"Huber": {"api": "manual.api"}}}
  torch_snap.write_text(json.dumps(torch_data))

  with patch("ml_switcheroo.semantics.autogen.resolve_snapshots_dir", return_value=snap_dir):
    persister.persist([sample_candidate], target_spec)

  # Verify Hub Protection
  new_hub = json.loads(target_spec.read_text())
  assert new_hub["Huber"]["std_args"] == ["manual"]  # Not ["reduction"]

  # Verify Torch Spoke Protection
  new_torch = json.loads(torch_snap.read_text())
  assert new_torch["mappings"]["Huber"]["api"] == "manual.api"  # Not "torch.nn.HuberLoss"

  # Verify JAX Spoke Creation (Additive)
  jax_snap = snap_dir / "jax_vlatest_map.json"
  assert jax_snap.exists()
  new_jax = json.loads(jax_snap.read_text())
  assert "Huber" in new_jax["mappings"]


def test_corrupt_file_handling(persister, sample_candidate, mock_fs):
  """
  Verify empty-dict fallback on corrupt JSON (robustness).
  """
  sem_dir, snap_dir = mock_fs
  target_spec = sem_dir / "corrupt.json"
  target_spec.write_text("{bad json")

  with patch("ml_switcheroo.semantics.autogen.resolve_snapshots_dir", return_value=snap_dir):
    # Should warning but not crash, then overwrite/repair
    persister.persist([sample_candidate], target_spec)

  assert target_spec.exists()
  data = json.loads(target_spec.read_text())
  assert "Huber" in data
