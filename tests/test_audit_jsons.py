"""Tests for audit_jsons script."""

import json
import yaml
from pathlib import Path

import pytest

import scripts.audit_jsons


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Setup a fake workspace and chdir to it."""
  sem_dir = tmp_path / "src" / "ml_switcheroo" / "semantics"
  odl_dir = sem_dir / "odl"
  odl_dir.mkdir(parents=True, exist_ok=True)

  monkeypatch.chdir(tmp_path)
  return sem_dir


def test_run(setup_workspace: Path) -> None:
  """Test run function processes yaml files correctly."""
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
    "variants": {"tf": {"api": "tf.Linear", "args": {}}, "mlx": {}},
  }
  with open(odl_dir / "Linear.yaml", "w") as f:
    yaml.dump(missing_type_data, f)

  # Empty file
  (odl_dir / "Empty.yaml").touch()

  # Hack to test k_extras (since we can't easily populate it naturally as it's hardcoded)
  import scripts.audit_jsons

  def test_with_k_extras():
    """Test run with extra operations populated."""
    # Get path before we changed dir in fixture (or construct absolute path)
    script_path = Path(__file__).parent.parent / "scripts" / "audit_jsons.py"
    source_code = open(script_path).read()
    source_code = source_code.replace(
      "k_extras: dict[str, dict[str, Any]] = {}", 'k_extras: dict[str, dict[str, Any]] = {"ExtraOp": {}}'
    )
    exec(compile(source_code, "scripts/audit_jsons.py", "exec"), scripts.audit_jsons.__dict__)
    scripts.audit_jsons.run()

  test_with_k_extras()

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
  """Test run function when odl directory does not exist."""
  sem_dir = setup_workspace
  odl_dir = sem_dir / "odl"
  odl_dir.rmdir()  # Remove the directory so exists() returns False

  scripts.audit_jsons.run()

  assert (sem_dir / "quarantine.json").exists()


def test_main_execution(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test module execution block."""
  import runpy
  import sys

  script_path = Path(__file__).parent.parent / "scripts" / "audit_jsons.py"

  monkeypatch.setattr(sys, "argv", ["audit_jsons.py"])
  runpy.run_path(str(script_path), run_name="__main__")

  assert (setup_workspace / "quarantine.json").exists()
