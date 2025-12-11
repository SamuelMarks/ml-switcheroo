"""
End-to-End Integration Test for Distributed Semantics.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.utils.migrator import SemanticMigrator
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig

LEGACY_MATH_JSON = {
  "Abs": {
    "description": "Absolute value",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
  }
}

LEGACY_TEMPLATES_JSON = {"torch": {"import": "import torch"}, "jax": {"import": "import jax"}}


@pytest.fixture
def legacy_env(tmp_path):
  root = tmp_path / "src"
  sem_dir = root / "semantics"
  snap_dir = root / "snapshots"

  sem_dir.mkdir(parents=True)

  (sem_dir / "k_array_api.json").write_text(json.dumps(LEGACY_MATH_JSON))
  (sem_dir / "k_test_templates.json").write_text(json.dumps(LEGACY_TEMPLATES_JSON))

  return sem_dir, snap_dir


def test_migration_and_execution_flow(legacy_env):
  sem_dir, snap_dir = legacy_env

  migrator = SemanticMigrator(semantics_path=sem_dir, snapshots_path=snap_dir)
  migrator.migrate(dry_run=False)

  assert (snap_dir / "torch_mappings.json").exists()
  assert (snap_dir / "jax_mappings.json").exists()

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=["torch", "jax"]):
        mgr = SemanticsManager()
        config = RuntimeConfig(source_framework="torch", target_framework="jax")
        engine = ASTEngine(semantics=mgr, config=config)

        defn = mgr.get_definition("torch.abs")
        assert defn is not None, "Failed to load torch.abs from overlay"

        code = "y = torch.abs(x)"
        result = engine.run(code)

        assert result.success
        assert "jax.numpy.abs(x)" in result.code
