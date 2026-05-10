"""
End-to-End Integration Test for Distributed Semantics.
"""

import json

import pytest


LEGACY_MATH_JSON = {
  "Abs": {
    "description": "Absolute value",
    "std_args": ["x"],
    "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
  }
}


@pytest.fixture
def legacy_env(tmp_path):
  """Function docstring."""
  root = tmp_path / "src"
  sem_dir = root / "semantics"
  snap_dir = root / "snapshots"

  sem_dir.mkdir(parents=True)

  (sem_dir / "k_array_api.json").write_text(json.dumps(LEGACY_MATH_JSON))

  return sem_dir, snap_dir
