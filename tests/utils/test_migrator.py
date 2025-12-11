"""
Tests for SemanticMigrator Utility.

Verifies that the tool:
1. Reads all JSONs in the semantics directory.
2. Correctly strips `variants` from the output spec files.
3. correctly creates `snapshots/{fw}_mappings.json` with extracted variants.
4. Moves `test_templates` into the corresponding framework mapping files.
5. Deletes obsolete files (like k_test_templates.json) after migration.
"""

import json
import pytest
from pathlib import Path

from ml_switcheroo.utils.migrator import SemanticMigrator


@pytest.fixture
def env_root(tmp_path):
  """
  Sets up a mock semantics directory structure.

  /semantics
     k_math.json
     k_test_templates.json
  /snapshots (Empty)
  """
  sem = tmp_path / "semantics"
  sem.mkdir()
  snap = tmp_path / "snapshots"
  snap.mkdir()

  # 1. Math Spec (With Variants)
  math_data = {
    "Abs": {
      "description": "Absolute Value",
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
    },
    "Add": {"std_args": ["a", "b"], "variants": {"numpy": {"api": "numpy.add"}}},
  }
  (sem / "k_math.json").write_text(json.dumps(math_data))

  # 2. Test Templates
  tmpl_data = {"torch": {"import": "import torch"}, "jax": {"import": "import jax"}}
  (sem / "k_test_templates.json").write_text(json.dumps(tmpl_data))

  return tmp_path


def test_migration_splits_spec_and_variants(env_root):
  """
  Scenario: Run migration on k_math.json.
  Expectation:
    - k_math.json loses 'variants' key.
    - snapshots/torch_mappings.json created with 'Abs'.
    - snapshots/jax_mappings.json created with 'Abs'.
    - snapshots/numpy_mappings.json created with 'Add'.
  """
  sem_dir = env_root / "semantics"
  snap_dir = env_root / "snapshots"

  migrator = SemanticMigrator(semantics_path=sem_dir, snapshots_path=snap_dir)
  migrator.migrate(dry_run=False)

  # 1. Check Spec File (Cleaned)
  with open(sem_dir / "k_math.json") as f:
    new_spec = json.load(f)

  assert "variants" not in new_spec["Abs"]
  assert new_spec["Abs"]["std_args"] == ["x"]
  assert new_spec["Add"]["std_args"] == ["a", "b"]

  # 2. Check Torch Overlay
  torch_file = snap_dir / "torch_mappings.json"
  assert torch_file.exists()

  with open(torch_file) as f:
    t_data = json.load(f)

  assert t_data["__framework__"] == "torch"
  assert "Abs" in t_data["mappings"]
  assert t_data["mappings"]["Abs"]["api"] == "torch.abs"
  # 'Add' had no torch variant, so it shouldn't be here
  assert "Add" not in t_data["mappings"]

  # 3. Check Numpy Overlay
  numpy_file = snap_dir / "numpy_mappings.json"
  assert numpy_file.exists()
  with open(numpy_file) as f:
    n_data = json.load(f)
  assert "Add" in n_data["mappings"]


def test_migration_moves_templates(env_root):
  """
  Scenario: k_test_templates.json exists.
  Expectation:
    - Content moved to 'templates' key in respective mapping files.
    - Original file deleted.
  """
  sem_dir = env_root / "semantics"
  snap_dir = env_root / "snapshots"

  migrator = SemanticMigrator(semantics_path=sem_dir, snapshots_path=snap_dir)
  migrator.migrate(dry_run=False)

  # 1. Check file deletion
  assert not (sem_dir / "k_test_templates.json").exists()

  # 2. Check injection into mapping file
  torch_file = snap_dir / "torch_mappings.json"
  with open(torch_file) as f:
    t_data = json.load(f)

  assert "templates" in t_data
  assert t_data["templates"]["import"] == "import torch"


def test_dry_run_safety(env_root):
  """
  Verify dry_run does not touch files.
  """
  sem_dir = env_root / "semantics"
  snap_dir = env_root / "snapshots"

  migrator = SemanticMigrator(semantics_path=sem_dir, snapshots_path=snap_dir)
  migrator.migrate(dry_run=True)

  # Files should remain unchanged
  assert (sem_dir / "k_test_templates.json").exists()
  assert not (snap_dir / "torch_mappings.json").exists()

  with open(sem_dir / "k_math.json") as f:
    data = json.load(f)
  assert "variants" in data["Abs"]
