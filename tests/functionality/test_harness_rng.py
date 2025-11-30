"""
Tests for RNG State Injection in Verification Harness.

Verifies that:
1. The harness detects when a target function signature has extra arguments
   (`rng`, `key`, `rngs`) compared to the source.
2. The harness automatically generates and injects valid JAX PRNG keys.
3. Execution succeeds even when signatures mismatch due to plugin injection.
"""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Runs the generated harness in a clean subprocess."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_rng_injection_jax(tmp_path):
  """
  Scenario:
      Source: def forward(x): ...
      Target: def forward(rng, x): ... (Transpiled with rng_threading plugin)
  Expecation:
      Harness detects 'rng' in target, creates jax.random.PRNGKey, and calls target.
  """
  # 1. Source (Torch-like, no RNG arg)
  src_file = tmp_path / "model_rng_src.py"
  src_file.write_text(""" 
import numpy as np
def forward(x): 
    # Deterministic op to match target logic for verification pass
    return x * 2
""")

  # 2. Target (JAX-like, injected RNG arg)
  tgt_file = tmp_path / "model_rng_tgt.py"
  tgt_file.write_text(""" 
import numpy as np
try: 
    import jax
    import jax.random
except ImportError: 
    pass

def forward(rng, x): 
    # Verify we got a valid PRNGKey or similar (e.g. fallback string in test env) 
    if rng is None: 
        raise ValueError("RNG argument is None!") 
    
    return x * 2
""")

  harness_path = tmp_path / "verify_rng.py"

  # 3. Generate
  gen = HarnessGenerator()
  gen.generate(
    src_file,
    tgt_file,
    harness_path,
    source_fw="numpy",  # Use numpy to allow basic fuzzer gen
    target_fw="jax",  # Target JAX to trigger logic check if needed
  )

  # 4. Execute
  result = _run_harness(harness_path)

  assert result.returncode == 0
  assert "✅ forward: Match" in result.stdout


def test_key_injection_alias(tmp_path):
  """
  Scenario: Target uses 'key' argument instead of 'rng'.
  """
  src_file = tmp_path / "model_key_src.py"
  src_file.write_text(""" 
def predict(x): return x
""")

  tgt_file = tmp_path / "model_key_tgt.py"
  tgt_file.write_text(""" 
def predict(key, x): 
    if key is None: 
        raise ValueError("Key missing") 
    return x
""")

  harness_path = tmp_path / "verify_key.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="jax")

  result = _run_harness(harness_path)

  assert "TypeError" not in result.stdout
  assert "✅ predict: Match" in result.stdout


def test_flax_rngs_injection(tmp_path):
  """
  Scenario: Target uses 'rngs' (Flax NNX pattern).
  """
  src_file = tmp_path / "model_nnx_src.py"
  src_file.write_text("def init(x): return x")

  tgt_file = tmp_path / "model_nnx_tgt.py"
  tgt_file.write_text(""" 
def init(rngs, x): 
    return x # Echo
""")

  harness_path = tmp_path / "verify_nnx.py"
  gen = HarnessGenerator()
  # Use numpy as source_fw/target_fw to ensure the harness uses the generic
  # adapter path (numpy array) instead of trying to import torch,
  # avoiding crashes in test environments where torch is missing.
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy")

  result = _run_harness(harness_path)

  # Should find match.
  # If this fails with Runtime Error "missing argument rngs", it means injection failed.
  assert "✅ init: Match" in result.stdout
