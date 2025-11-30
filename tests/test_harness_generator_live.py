"""
Integration test for the Harness Generator using live code extraction.

Verifies that:
1. The generator correctly extracts `InputFuzzer` at runtime.
2. The shim for `get_adapter` is injected.
3. The resulting script executes and the embedded Fuzzer logic runs.
"""

import sys
import subprocess
import os
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def test_generated_fuzzer_runs_standalone(tmp_path):
  """
  Verifies that the injected InputFuzzer logic works without 'ml_switcheroo' in path.
  """
  src = tmp_path / "src.py"
  tgt = tmp_path / "tgt.py"
  # Simple identity function
  src.write_text("def f(x): return x")
  tgt.write_text("def f(x): return x")

  harness = tmp_path / "verify_live.py"

  # Generate
  gen = HarnessGenerator()
  gen.generate(src, tgt, harness, source_fw="numpy", target_fw="numpy")

  # Read output to inspect content
  content = harness.read_text()
  assert "class InputFuzzer" in content
  # Check for the Shim injection
  assert "def get_adapter(framework):" in content
  assert "GenericAdapter" in content

  # Run Standalone (Strip PYTHONPATH)
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  # We expect success because logic is synced from real fuzzer
  res = subprocess.run([sys.executable, str(harness)], capture_output=True, text=True, env=env)

  if res.returncode != 0:
    print("STDERR:", res.stderr)

  assert res.returncode == 0
  assert "✅ f: Match" in res.stdout
