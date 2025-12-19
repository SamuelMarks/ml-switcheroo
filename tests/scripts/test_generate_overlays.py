"""
Tests for the Unified Knowledge Base Generator.

Verifies that `generate_overlays.py` correctly populates the semantics
and snapshots directory with valid JSON content covering key scenarios.
"""

import json
import pytest
from pathlib import Path

# Import the main function from the script module
# We need to construct the module path to import it since it's outside src/
import importlib.util
import sys

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "generate_overlays.py"


def load_generator_module():
  spec = importlib.util.spec_from_file_location("generate_overlays", SCRIPT_PATH)
  mod = importlib.util.module_from_spec(spec)
  sys.modules["generate_overlays"] = mod
  spec.loader.exec_module(mod)
  return mod


generator = load_generator_module()


@pytest.fixture
def mock_fs(tmp_path):
  """
  Sets up a temporary filesystem and patches the generator's paths
  to write to this temp location instead of the real src/.
  """
  sem = tmp_path / "semantics"
  snap = tmp_path / "snapshots"
  sem.mkdir()
  snap.mkdir()

  # Patch module global constants
  generator.SEMANTICS_DIR = sem
  generator.SNAPSHOTS_DIR = snap
  generator.ROOT_DIR = tmp_path

  return sem, snap


def test_hub_spec_generation(mock_fs):
  """
  Verify that Abstract Specs (Hub) are created and contain required keys.
  """
  sem_dir, _ = mock_fs

  # Run Generation
  generator.update_specs()

  # Check Neural Spec
  nn_file = sem_dir / "k_neural_net.json"
  assert nn_file.exists()
  nn_data = json.loads(nn_file.read_text())

  assert "BatchNorm" in nn_data
  assert nn_data["BatchNorm"]["std_args"] == ["input", "eps"]

  # Check Array Spec
  arr_file = sem_dir / "k_array_api.json"
  assert arr_file.exists()
  arr_data = json.loads(arr_file.read_text())

  assert "Clamp" in arr_data
  assert "CastFloat" in arr_data


def test_spoke_mapping_generation(mock_fs):
  """
  Verify that Framework Mappings (Spokes) are created with correct plugin wiring.
  """
  _, snap_dir = mock_fs

  # Run Overlay Generators
  generator.gen_torch_mappings()
  generator.gen_jax_mappings()
  generator.gen_mlx_mappings()

  # 1. Check Torch
  t_file = snap_dir / "torch_vlatest_map.json"
  assert t_file.exists()
  t_data = json.loads(t_file.read_text())

  assert t_data["mappings"]["BatchNorm"]["api"] == "torch.nn.BatchNorm2d"
  assert t_data["mappings"]["Clamp"]["api"] == "torch.clamp"
  # Check functional transform pivot
  assert t_data["mappings"]["vmap"]["args"]["in_axes"] == "in_dims"

  # 2. Check JAX
  j_file = snap_dir / "jax_vlatest_map.json"
  assert j_file.exists()
  j_data = json.loads(j_file.read_text())

  # Check Plugin Wiring
  assert j_data["mappings"]["BatchNorm"]["requires_plugin"] == "batch_norm_unwrap"
  assert j_data["mappings"]["Gather"]["requires_plugin"] == "gather_adapter"
  assert j_data["mappings"]["ClipGradNorm"]["requires_plugin"] == "grad_clipper"

  # Check Argument Pivots
  assert j_data["mappings"]["Clamp"]["args"]["min"] == "a_min"

  # 3. Check MLX
  m_file = snap_dir / "mlx_vlatest_map.json"
  assert m_file.exists()
  m_data = json.loads(m_file.read_text())

  assert m_data["mappings"]["Compile"]["api"] == "mlx.core.compile"


def test_merge_behavior(mock_fs):
  """
  Verify that running the generator merges with existing data instead of overwriting.
  """
  _, snap_dir = mock_fs

  # Pre-seed a file with custom user data
  t_file = snap_dir / "torch_vlatest_map.json"
  t_file.write_text(json.dumps({"mappings": {"CustomUserOp": {"api": "user.custom"}}}))

  # Run Generator
  generator.gen_torch_mappings()

  # Check Result
  t_data = json.loads(t_file.read_text())

  # Generated content should exist
  assert "BatchNorm" in t_data["mappings"]
  # User content should be preserved
  assert "CustomUserOp" in t_data["mappings"]
