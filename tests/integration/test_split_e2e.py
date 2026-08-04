"""Integration and E2E test suite verifying split semantics environments.

This module sets up integration and end-to-end (E2E) testing configurations for the
split operations pipeline. It provides mock metadata representation of legacy operators
and a pytest fixture to prepare temporary directory layouts that simulate standard
project file hierarchies for semantics configuration and snapshots.
"""

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
  """Provides a temporary, mock legacy environment structure for integration testing.

  This fixture prepares the directory structure representing the semantics and
  snapshots location required by the conversion engines. It creates an 'odl'
  subdirectory inside the 'semantics' directory and populates it with ODL YAML
  specifications matching the registered legacy mathematical operations.

  Args:
      tmp_path (pathlib.Path): A pytest-supplied temporary directory path
          unique to each test invocation.

  Returns:
      tuple[pathlib.Path, pathlib.Path]: A tuple containing two paths:
          - sem_dir (pathlib.Path): Path to the semantics configuration directory.
          - snap_dir (pathlib.Path): Path to the snapshots comparison directory.
  """
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
