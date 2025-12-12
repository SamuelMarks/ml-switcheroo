"""
End-to-End Integration Test for Distributed Semantics.
"""

import json
from unittest.mock import patch

import pytest

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


@pytest.fixture
def legacy_env(tmp_path):
  root = tmp_path / "src"
  sem_dir = root / "semantics"
  snap_dir = root / "snapshots"

  sem_dir.mkdir(parents=True)

  (sem_dir / "k_array_api.json").write_text(json.dumps(LEGACY_MATH_JSON))

  return sem_dir, snap_dir
