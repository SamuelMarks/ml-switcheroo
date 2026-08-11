"""Tests for scripts/generate_10k_plan.py."""

import sys
import json
from pathlib import Path
from unittest import mock
import pytest

# Add scripts directory to sys.path to import it
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir.resolve()))

import generate_10k_plan  # noqa: E402


@pytest.fixture
def mock_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  """Sets up mock repositories and temporary directories."""
  tmp_repos = tmp_path / "tmp_repos"
  monkeypatch.setattr(generate_10k_plan, "TMP_DIR", tmp_repos)

  mock_repos = {
    "pytorch": "https://pytorch.com",
    "keras": "https://keras.com",
    "flax": "https://flax.com",
    "mlx": "https://mlx.com",
    "jax": "https://jax.com",
    "numpy": "https://numpy.com",
    "transformers": "https://transformers.com",
    "maxtext": "https://maxtext.com",
  }
  monkeypatch.setattr(generate_10k_plan, "REPOS", mock_repos)

  mock_dirs = {k: ["core"] for k in mock_repos}
  monkeypatch.setattr(generate_10k_plan, "FOCUS_DIRS", mock_dirs)

  return tmp_repos


def test_clone_repos(mock_env: Path, monkeypatch: pytest.MonkeyPatch):
  """Test repository cloning logic."""
  # Create one existing repo to test skip path
  repo_a = mock_env / "pytorch"
  repo_a.mkdir(parents=True)

  mock_run = mock.Mock()
  monkeypatch.setattr(generate_10k_plan.subprocess, "run", mock_run)

  generate_10k_plan.clone_repos()

  # 8 total repos, 1 skipped, so 7 cloned
  assert mock_run.call_count == 7


def test_extract_api_surface_file_and_dir(mock_env: Path):
  """Test API extraction for both files and directories."""
  repo_a = mock_env / "pytorch"

  # Set up a directory with a Python file and a test file
  layers_dir = repo_a / "core"
  layers_dir.mkdir(parents=True)
  (layers_dir / "dense.py").write_text("class Dense:\n  pass\nclass _PrivateDense:\n  pass\n", encoding="utf-8")
  (layers_dir / "test_dense.py").write_text("class TestDense:\n  pass", encoding="utf-8")

  functional_file = layers_dir / "functional.py"
  functional_file.write_text(
    "def relu():\n  pass\ndef _private():\n  pass\nclass Outer:\n  def inner_method(self):\n    pass\n",
    encoding="utf-8",
  )

  apis = generate_10k_plan.extract_api_surface("pytorch", ["core", "core/functional.py"])

  # Expected: dense.Dense, functional.Outer, functional.relu (inner_method skipped, private skipped, test skipped)
  assert apis == ["dense.Dense", "functional.Outer", "functional.relu"]


def test_extract_api_surface_parse_error(mock_env: Path):
  """Test API extraction handles parse errors gracefully."""
  repo_b = mock_env / "keras"
  core_dir = repo_b / "core"
  core_dir.mkdir(parents=True)

  bad_file = core_dir / "bad.py"
  bad_file.write_text("class InvalidSyntax(", encoding="utf-8")

  apis = generate_10k_plan.extract_api_surface("keras", ["core"])
  assert apis == []


def test_generate_mappings(mock_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  """Test the mapping generation logic."""

  # Mock extract_api_surface to return predetermined APIs
  def mock_extract(repo_name, focus_dirs):
    """Mock extract."""
    if repo_name == "pytorch":
      return ["module.Dense", "module.Relu"]
    elif repo_name == "keras":
      return ["other.dense", "other.Softmax"]
    return ["dummy.Op"]

  monkeypatch.setattr(generate_10k_plan, "extract_api_surface", mock_extract)

  plan_path = tmp_path / "10000_STEP_PLAN.md"
  json_path = tmp_path / "universal_mapping.json"

  original_open = Path.open

  def mock_open(self, *args, **kwargs):
    """Mock open."""
    if self.name == "10000_STEP_PLAN.md":
      return original_open(plan_path, *args, **kwargs)
    elif self.name == "universal_mapping.json":
      return original_open(json_path, *args, **kwargs)
    return original_open(self, *args, **kwargs)

  monkeypatch.setattr(Path, "open", mock_open)

  generate_10k_plan.generate_mappings()

  # Check json output
  assert json_path.exists()
  with open(json_path) as f:
    mappings = json.load(f)

  assert "pytorch_to_keras" in mappings
  pt_to_kr = mappings["pytorch_to_keras"]

  assert pt_to_kr["module.Dense"]["type"] == "direct"
  assert pt_to_kr["module.Dense"]["target"] == "other.dense"

  assert pt_to_kr["module.Relu"]["type"] == "decompose"
  assert pt_to_kr["module.Relu"]["intermediate"] == "jax"


def test_main_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  """Test the main execution block."""
  import runpy
  import subprocess
  import os

  monkeypatch.setattr(subprocess, "run", mock.Mock())

  original_cwd = os.getcwd()
  os.chdir(tmp_path)
  try:
    runpy.run_path(str(scripts_dir / "generate_10k_plan.py"), run_name="__main__")

    # Assert it created the expected output files
    assert (tmp_path / "10000_STEP_PLAN.md").exists()
    assert (tmp_path / "universal_mapping.json").exists()
  finally:
    os.chdir(original_cwd)
