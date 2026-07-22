"""Test suite for the Harness module."""

import sys
import subprocess
import os
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def test_harness_generation_file_creation(tmp_path):
  """Verifies the behavior of harness generation file creation."""
  src_file = tmp_path / "source.py"
  src_file.write_text("def fn(x): return x")
  tgt_file = tmp_path / "target.py"
  tgt_file.write_text("def fn(x): return x")
  harness_path = tmp_path / "verify.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path)
  assert harness_path.exists()
  content = harness_path.read_text()
  assert "class InputFuzzer" in content
  assert "class StandaloneFuzzer" in content
  assert "from ml_switcheroo.testing.fuzzer" not in content


def test_harness_execution_standalone(tmp_path):
  """Verifies the behavior of harness execution standalone."""
  src_file = tmp_path / "mod_src.py"
  src_file.write_text("\nimport numpy as np\ndef my_op(x):\n    return x * 2\n")
  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text("\nimport numpy as np\ndef my_op(x):\n    return x * 2\n")
  harness_path = tmp_path / "verify_match.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy")
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  result = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True, env=env)
  assert result.returncode == 0
  assert "✅ my_op: Match" in result.stdout


def test_harness_execution_mismatch(tmp_path):
  """Verifies the behavior of harness execution mismatch."""
  src_file = tmp_path / "mod_src.py"
  src_file.write_text("def my_op(x): return x + 1")
  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text("def my_op(x): return x + 100")
  harness_path = tmp_path / "verify_fail.py"
  gen = HarnessGenerator()
  semantics = {"my_op": {"std_args": [{"name": "x", "type": "float"}]}}
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy", semantics=semantics)
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  result = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True, env=env)
  assert result.returncode == 1
  assert "❌ my_op: Mismatch" in result.stdout
