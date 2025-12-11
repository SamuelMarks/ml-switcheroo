"""
Tests for Semantic Persistence (Autogen).

Verifies:
1. File Creation: New files are created if missing.
2. Format Compliance: CandidateStandard is correctly mapped to JSON schema.
3. Conflict Resolution: Existing keys (Manual Overrides) are protected.
4. IO Safety: Handle corrupt JSON gracefully.
"""

import json
import pytest
from pathlib import Path
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.discovery.consensus import CandidateStandard
from ml_switcheroo.core.ghost import GhostRef


# --- Fixtures ---


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
  )
  return c


# --- Tests ---


def test_persist_new_file_creation(persister, sample_candidate, tmp_path):
  """Scenario: target file does not exist."""
  target = tmp_path / "k_generated.json"

  persister.persist([sample_candidate], target)

  assert target.exists()

  data = json.loads(target.read_text("utf-8"))

  # Check Structure
  assert "Huber" in data
  entry = data["Huber"]

  assert "variants" in entry
  assert "std_args" in entry
  assert entry["std_args"] == ["reduction"]

  # Check Variants
  assert entry["variants"]["torch"]["api"] == "torch.nn.HuberLoss"
  assert entry["variants"]["jax"]["args"]["reduction"] == "reduction_mode"


def test_persist_protects_manual_overrides(persister, sample_candidate, tmp_path):
  """
  Scenario: target file exists and contains 'Huber'.
  Expectation: The existing 'Huber' entry is NOT touched.
  """
  target = tmp_path / "k_overrides.json"

  # Simulate a Manual Override (different from auto)
  manual_data = {"Huber": {"variants": {}, "std_args": ["manual_flag"], "doc": "Manually Curated"}}
  target.write_text(json.dumps(manual_data), encoding="utf-8")

  persister.persist([sample_candidate], target)

  # Reload and Verify
  data = json.loads(target.read_text("utf-8"))

  assert data["Huber"]["doc"] == "Manually Curated"
  assert "torch" not in data["Huber"]["variants"]
  assert "manual_flag" in data["Huber"]["std_args"]


def test_persist_merges_disjoint_sets(persister, sample_candidate, tmp_path):
  """
  Scenario: File exists with 'MSE', new candidate is 'Huber'.
  Expectation: 'MSE' preserved, 'Huber' added.
  """
  target = tmp_path / "k_mixed.json"

  existing = {"MSE": {"variants": {"f": "v"}}}
  target.write_text(json.dumps(existing), encoding="utf-8")

  persister.persist([sample_candidate], target)

  data = json.loads(target.read_text("utf-8"))

  assert "MSE" in data
  assert "Huber" in data


def test_handle_corrupt_json(persister, sample_candidate, tmp_path):
  """
  Scenario: Target file contains garbage.
  Expectation: File is backed up, logic starts fresh/overwrites to recover.
  """
  target = tmp_path / "corrupt.json"
  target.write_text("{NOT VALID JSON", encoding="utf-8")

  persister.persist([sample_candidate], target)

  # Should have created .bak
  assert (target.with_suffix(".bak")).exists()

  # Should have written valid new file
  data = json.loads(target.read_text("utf-8"))
  assert "Huber" in data
