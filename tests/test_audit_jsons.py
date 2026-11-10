"""Tests for audit_jsons script."""

import json
from pathlib import Path

import pytest
import yaml

import scripts.audit_jsons


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Setup a fake workspace and chdir to it.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.

  Returns:
      Configured `setup_workspace` fixture value (Path).
  """
  sem_dir = tmp_path / "src" / "ml_switcheroo" / "semantics"
  odl_dir = sem_dir / "odl"
  odl_dir.mkdir(parents=True, exist_ok=True)

  monkeypatch.chdir(tmp_path)
  return sem_dir


def test_run(setup_workspace: Path) -> None:
  """Test run function processes yaml files correctly.

  Args:
      setup_workspace: The setup workspace (Path) for test run.
  """
  sem_dir = setup_workspace
  odl_dir = sem_dir / "odl"

  # Create valid file
  valid_data = {
    "operation": "Conv2d",
    "std_args": ["arg1", {"name": "arg2"}],
    "variants": {"torch": {"api": "torch.nn.Conv2d"}},
  }
  with open(odl_dir / "Conv2d.yaml", "w") as f:
    yaml.dump(valid_data, f)

  # Create invalid file (broken YAML)
  with open(odl_dir / "Broken.yaml", "w") as f:
    f.write("Broken:\n- YAML\n  - Error: [\n")

  # Create unverified/quarantined file
  unverified_data = {"operation": "UnknownOp", "std_args": [], "variants": {}}
  with open(odl_dir / "UnknownOp.yaml", "w") as f:
    yaml.dump(unverified_data, f)

  # Create file with missing name and type in args and missing variants keys
  missing_type_data = {
    "operation": "Linear",
    "std_args": [{"other": "val"}, "str_arg", 42, {"name": "has_name", "type": "has_type"}],
    "variants": {"tf": {"api": "tf.Linear", "args": {}}, "mlx": {}, "bad_var": "not_a_dict"},
  }
  with open(odl_dir / "Linear.yaml", "w") as f:
    yaml.dump(missing_type_data, f)

  # Create verified op with non-list std_args and non-dict variants
  non_list_data = {
    "operation": "Dense",
    "std_args": "not_a_list",
    "variants": "not_a_dict",
  }
  with open(odl_dir / "Dense.yaml", "w") as f:
    yaml.dump(non_list_data, f)

  # Empty file
  (odl_dir / "Empty.yaml").touch()

  scripts.audit_jsons.run(extra_ops={"ExtraOp": {}})

  # Verify outputs
  assert (sem_dir / "quarantine.json").exists()

  with open(sem_dir / "quarantine.json") as f:
    quarantine = json.load(f)
  assert "UnknownOp" in quarantine

  with open(odl_dir / "Conv2d.yaml") as f:
    rewritten = yaml.safe_load(f)
  assert rewritten["std_args"][0] == {"name": "arg1", "type": "Any"}
  assert rewritten["std_args"][1] == {"name": "arg2", "type": "Any"}

  with open(odl_dir / "Linear.yaml") as f:
    rewritten_lin = yaml.safe_load(f)
  assert rewritten_lin["std_args"][0]["name"] == "unknown"
  assert rewritten_lin["std_args"][0]["type"] == "Any"
  assert rewritten_lin["std_args"][1] == {"name": "str_arg", "type": "Any"}
  assert rewritten_lin["std_args"][2] == 42
  assert rewritten_lin["variants"]["tf"]["api"] == "tf.Linear"
  assert rewritten_lin["variants"]["tf"]["args"] == {}


def test_run_no_odl_dir(setup_workspace: Path) -> None:
  """Test run function when odl directory does not exist.

  Args:
      setup_workspace: The setup workspace (Path) for test run no odl dir.
  """
  sem_dir = setup_workspace
  odl_dir = sem_dir / "odl"
  odl_dir.rmdir()  # Remove the directory so exists() returns False

  scripts.audit_jsons.run()

  assert (sem_dir / "quarantine.json").exists()


def test_main_execution(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test module execution block.

  Args:
      setup_workspace: The setup workspace (Path) for test main execution.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  import runpy
  import sys

  script_path = Path(__file__).parent.parent / "scripts" / "audit_jsons.py"

  monkeypatch.setattr(sys, "argv", ["audit_jsons.py"])
  runpy.run_path(str(script_path), run_name="__main__")

  assert (setup_workspace / "quarantine.json").exists()
