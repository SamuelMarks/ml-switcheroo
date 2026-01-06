"""
Tests for verify functionality (Harness Generation).
Checks that the harness is generated correctly, compiles, and runs standalone.
"""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def test_harness_generation_file_creation(tmp_path):
  """
  Verify the harness file is physically created and standalone.
  """
  src_file = tmp_path / "source.py"
  src_file.write_text("def fn(x): return x")

  tgt_file = tmp_path / "target.py"
  tgt_file.write_text("def fn(x): return x")

  harness_path = tmp_path / "verify.py"

  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path)

  assert harness_path.exists()
  content = harness_path.read_text()

  # Check Standalone Nature
  # We extract InputFuzzer logic, so verify class presence
  assert "class InputFuzzer" in content
  assert "class StandaloneFuzzer" in content

  # Ensure dependencies are removed
  assert "from ml_switcheroo.testing.fuzzer" not in content


def test_harness_execution_standalone(tmp_path):
  """
  Verify correctness by running the script in a subprocess WITHOUT adding
  ml_switcheroo to PYTHONPATH. This proves it is portable.
  """
  # 1. Source
  src_file = tmp_path / "mod_src.py"
  src_file.write_text(""" 
import numpy as np
def my_op(x): 
    return x * 2
""")

  # 2. Target
  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text(""" 
import numpy as np
def my_op(x): 
    return x * 2
""")

  harness_path = tmp_path / "verify_match.py"

  gen = HarnessGenerator()
  # Use 'numpy' as FW to use the basic path in GenericAdapter without torch/jax installed
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy")

  # 3. Clean Environment (Remove ml-switcheroo from Path)
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  # 4. Run
  result = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True, env=env)

  assert result.returncode == 0
  assert "✅ my_op: Match" in result.stdout


def test_harness_execution_mismatch(tmp_path):
  """
  Verify that logic failures are still caught in the standalone script.
  Forces float input type so generated test does not use empty arrays, ensuring numeric mismatch logic triggers.
  """
  src_file = tmp_path / "mod_src.py"
  src_file.write_text("def my_op(x): return x + 1")

  tgt_file = tmp_path / "mod_tgt.py"
  tgt_file.write_text("def my_op(x): return x + 100")

  harness_path = tmp_path / "verify_fail.py"

  gen = HarnessGenerator()
  # Semantics definition forces x to be float scalar, preventing empty array edge case
  semantics = {"my_op": {"std_args": [{"name": "x", "type": "float"}]}}

  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy", semantics=semantics)

  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  result = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True, env=env)

  assert result.returncode == 1
  assert "❌ my_op: Mismatch" in result.stdout
