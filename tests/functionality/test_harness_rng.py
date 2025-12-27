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
from unittest.mock import patch, MagicMock
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Runs the generated harness in a clean subprocess."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_rng_injection_jax(mock_get_adapter, tmp_path):
  """
  Scenario:
      Source: def forward(x): ...
      Target: def forward(rng, x): ... (Transpiled with rng_threading plugin)
  Expectation:
      Harness detects 'rng' in target, creates jax.random.PRNGKey, and calls target.
  """
  # Force use of JAX adapter which defines declares_magic_args=['key']
  # We use a mocked version that avoids import errors
  adapter = JaxCoreAdapter()

  # We force 'rng' into magic args for this test case specifically
  # Normally JAX uses 'key', but standardizing on 'rng' for this test logic
  with patch.object(JaxCoreAdapter, "declared_magic_args", ["rng"]):
    mock_get_adapter.return_value = adapter

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
    # We manually implement the helper function in the mock target because harness
    # injection is what we are testing, specifically the invocation of _make_jax_key
    tgt_file.write_text("""
import numpy as np
try: 
    import jax
    import jax.random
except ImportError: 
    pass

def forward(rng, x): 
    # Verify we got a valid PRNGKey or similar (e.g. fallback string in test env)
    # The default impl of _make_jax_key returns a mock string if import fails. 
    if rng is None: 
        raise ValueError("RNG argument is None!") 
    
    if rng == "mock_jax_key" or hasattr(rng, 'tolist'): # Check valid key
         return x * 2
    
    raise ValueError(f"Received invalid rng: {rng}") 
""")

    harness_path = tmp_path / "verify_rng.py"

    # 3. Generate with target="jax" to trigger the logic
    gen = HarnessGenerator()
    gen.generate(
      src_file,
      tgt_file,
      harness_path,
      source_fw="numpy",
      target_fw="jax",  # Trigger JAX logic
    )

    # 4. Execute
    result = _run_harness(harness_path)

    if result.returncode != 0:
      print("STDERR:", result.stderr)
      print("STDOUT:", result.stdout)

    assert result.returncode == 0
    assert "✅ forward: Match" in result.stdout


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_key_injection_alias(mock_get_adapter, tmp_path):
  """
  Scenario: Target uses 'key' argument instead of 'rng'.
  """
  adapter = JaxCoreAdapter()
  # 'key' is standard for JAX adapter
  mock_get_adapter.return_value = adapter

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

  # Use JAX target
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="jax")

  result = _run_harness(harness_path)

  assert "TypeError" not in result.stdout
  assert "✅ predict: Match" in result.stdout


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_flax_rngs_injection(mock_get_adapter, tmp_path):
  """
  Scenario: Target uses 'rngs' (Flax NNX pattern).
  """
  adapter = FlaxNNXAdapter()
  mock_get_adapter.return_value = adapter

  src_file = tmp_path / "model_nnx_src.py"
  src_file.write_text("def init(x): return x")

  tgt_file = tmp_path / "model_nnx_tgt.py"
  tgt_file.write_text("""
def init(rngs, x): 
    return x # Echo
""")

  harness_path = tmp_path / "verify_nnx.py"
  gen = HarnessGenerator()

  # Use Flax target
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="flax_nnx")

  result = _run_harness(harness_path)

  assert "✅ init: Match" in result.stdout
