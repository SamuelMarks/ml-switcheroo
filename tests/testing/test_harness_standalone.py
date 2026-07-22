"""Test suite for the Harness Standalone module."""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.frameworks import register_framework


class DynKerasAdapter:
  """Test suite for the Dyn Keras Adapter component."""

  declared_magic_args = []
  harness_imports = []

  def get_harness_init_code(self):
    """Gets harness initialization code."""
    return ""

  def convert(self, data):
    """Converts ."""
    return "KerasMock(" + str(data) + ")"


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Helper to  run harness."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_dynamic_shim_generation(tmp_path):
  """Verifies the behavior of dynamic shim generation."""
  register_framework("mock_keras")(DynKerasAdapter)
  gen = HarnessGenerator()
  harness_path = tmp_path / "verify_shim.py"
  gen.generate(tmp_path, tmp_path, harness_path)
  content = harness_path.read_text()
  assert "if framework == 'mock_keras':" in content or "elif framework == 'mock_keras':" in content
  assert 'return "KerasMock(" + str(data) + ")"' in content


def test_harness_execution_match(tmp_path):
  """Verifies the behavior of harness execution match."""
  src_file = tmp_path / "mod_src.py"
  src_file.write_text("def my_op(x): return x * 2")
  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text("def my_op(x): return x * 2")
  harness_path = tmp_path / "verify_match.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy")
  result = _run_harness(harness_path)
  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
  assert result.returncode == 0
  assert "✅ my_op: Match" in result.stdout
