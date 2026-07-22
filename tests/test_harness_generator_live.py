"""Test suite for the Harness Generator Live module."""

import sys
import subprocess
import os
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def test_generated_fuzzer_runs_standalone(tmp_path):
  """Verifies the behavior of generated fuzzer runs standalone."""
  src = tmp_path / "src.py"
  tgt = tmp_path / "tgt.py"
  src.write_text("def f(x): return x")
  tgt.write_text("def f(x): return x")
  harness = tmp_path / "verify_live.py"
  gen = HarnessGenerator()
  gen.generate(src, tgt, harness, source_fw="numpy", target_fw="numpy")
  content = harness.read_text()
  assert "class InputFuzzer" in content
  assert "def get_adapter(framework):" in content
  assert "GenericAdapter" in content
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  res = subprocess.run([sys.executable, str(harness)], capture_output=True, text=True, env=env)
  if res.returncode != 0:
    print("STDERR:", res.stderr)
  assert res.returncode == 0
  assert "✅ f: Match" in res.stdout
