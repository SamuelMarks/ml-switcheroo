"""
Integration Tests for Standalone Harness Verification (Dynamic Shim).

Verifies that:
1. The HarnessGenerator correctly extracts `InputFuzzer` logic.
2. It dynamically builds the `get_adapter` shim from registered frameworks.
3. The generated script can execute without `ml_switcheroo` in PYTHONPATH.
"""

import sys
import subprocess
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.frameworks import register_framework


# --- Mock Adapter for Dynamic Test ---
class DynKerasAdapter:
  """Mock Keras Adapter with specific convert logic."""

  declared_magic_args = []
  harness_imports = []

  def get_harness_init_code(self):
    return ""

  def convert(self, data):
    return "KerasMock(" + str(data) + ")"


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Runs the generated harness in a clean subprocess environment."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_dynamic_shim_generation(tmp_path):
  """
  Verify that the shim includes logic from a dynamically registered adapter.
  """
  # 1. Register a custom framework
  register_framework("mock_keras")(DynKerasAdapter)

  # 2. Generate Harness
  gen = HarnessGenerator()
  harness_path = tmp_path / "verify_shim.py"

  # Dummy paths
  gen.generate(tmp_path, tmp_path, harness_path)

  content = harness_path.read_text()

  # 3. Verify Shim Logic
  assert "if framework == 'mock_keras':" in content or "elif framework == 'mock_keras':" in content

  # Verify extraction of converting logic
  assert 'return "KerasMock(" + str(data) + ")"' in content


def test_harness_execution_match(tmp_path):
  """
  Verify correctness: Identical functions return Exit Code 0.
  Using 'numpy' mode to avoid needing installed Torch/JAX in test worker.
  """
  src_file = tmp_path / "mod_src.py"
  src_file.write_text("def my_op(x): return x * 2")

  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text("def my_op(x): return x * 2")

  harness_path = tmp_path / "verify_match.py"
  gen = HarnessGenerator()

  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy")

  # Execute Standalone
  result = _run_harness(harness_path)

  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)

  assert result.returncode == 0
  assert "✅ my_op: Match" in result.stdout
