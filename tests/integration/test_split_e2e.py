"""Test suite for the Split E2E module."""

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
  """Provides a mock legacy environment for testing."""
  root = tmp_path / "src"
  sem_dir = root / "semantics"
  snap_dir = root / "snapshots"
  sem_dir.mkdir(parents=True)
  import yaml

  odl_dir = sem_dir / "odl"
  odl_dir.mkdir()
  for k, v in LEGACY_MATH_JSON.items():
    v["operation"] = k
    (odl_dir / f"{k.replace('/', '_')}.yaml").write_text(yaml.dump(v))
  return (sem_dir, snap_dir)
