"""Test suite for the Harness Generator Live module."""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from typing import Dict


def test_generated_fuzzer_runs_standalone(tmp_path: Path) -> None:
  """Verifies the behavior of generated fuzzer runs standalone."""
  src: Path = tmp_path / "src.py"
  tgt: Path = tmp_path / "tgt.py"
  src.write_text("def f(x): return x")
  tgt.write_text("def f(x): return x")
  harness: Path = tmp_path / "verify_live.py"
  gen: HarnessGenerator = HarnessGenerator()
  gen.generate(src, tgt, harness, source_fw="numpy", target_fw="numpy")
  content: str = harness.read_text()
  assert "class InputFuzzer" in content
  assert "def get_adapter(framework: str) -> typing.Any:" in content
  assert "GenericAdapter" in content
  env: Dict[str, str] = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  res: subprocess.CompletedProcess[str] = subprocess.run(
    [sys.executable, str(harness)], capture_output=True, text=True, env=env
  )
  if res.returncode != 0:
    print("STDERR:", res.stderr)
  assert res.returncode == 0
  assert "✅ f: Match" in res.stdout
