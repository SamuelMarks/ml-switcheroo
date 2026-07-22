"""Test suite for the Harness Rng module."""

import sys
import subprocess
import os
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Helper to  run harness."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


class SafeJaxCoreAdapter(JaxCoreAdapter):
  """Test suite for the Safe Jax Core Adapter component."""

  @property
  def harness_imports(self):
    """Helper to harness imports."""
    return []

  def get_harness_init_code(self):
    """Gets harness initialization code."""
    return '\ndef _make_jax_key(seed):\n    return "mock_jax_key"\n'


class SafeFlaxAdapter(FlaxNNXAdapter):
  """Test suite for the Safe Flax Adapter component."""

  @property
  def harness_imports(self):
    """Helper to harness imports."""
    return []

  def get_harness_init_code(self):
    """Gets harness initialization code."""
    return '\ndef _make_flax_rngs(seed):\n    return "mock_flax_rngs"\n'


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_rng_injection_jax(mock_get_adapter, tmp_path):
  """Verifies the behavior of rng injection JAX."""
  adapter = SafeJaxCoreAdapter()
  with patch.object(SafeJaxCoreAdapter, "declared_magic_args", ["rng"]):
    mock_get_adapter.return_value = adapter
    src_file = tmp_path / "model_rng_src.py"
    src_file.write_text(
      "\nimport numpy as np\ndef forward(x):\n    # Deterministic op to match target logic for verification pass\n    return x * 2\n"
    )
    tgt_file = tmp_path / "model_rng_tgt.py"
    tgt_file.write_text(
      '\nimport numpy as np\n# Safe imports for target mock file\ntry:\n    import jax\n    import jax.random\nexcept ImportError:\n    pass\n\ndef forward(rng, x):\n    # Verify we got a valid PRNGKey or similar (e.g. fallback string in test env)\n    # The default impl of _make_jax_key returns a mock string if import fails.\n    if rng is None:\n        raise ValueError("RNG argument is None!")\n\n    if rng == "mock_jax_key" or hasattr(rng, \'tolist\'): # Check valid key\n         return x * 2\n\n    raise ValueError(f"Received invalid rng: {rng}")\n'
    )
    harness_path = tmp_path / "verify_rng.py"
    gen = HarnessGenerator()
    gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="jax")
    result = _run_harness(harness_path)
    if result.returncode != 0:
      print("STDERR:", result.stderr)
      print("STDOUT:", result.stdout)
    assert result.returncode == 0
    assert "✅ forward: Match" in result.stdout


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_key_injection_alias(mock_get_adapter, tmp_path):
  """Verifies the behavior of key injection alias."""
  adapter = SafeJaxCoreAdapter()
  mock_get_adapter.return_value = adapter
  src_file = tmp_path / "model_key_src.py"
  src_file.write_text("\ndef predict(x): return x\n")
  tgt_file = tmp_path / "model_key_tgt.py"
  tgt_file.write_text(
    '\ndef predict(key, x):\n    if key is None:\n        raise ValueError("Key missing")\n    return x\n'
  )
  harness_path = tmp_path / "verify_key.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="jax")
  result = _run_harness(harness_path)
  assert "TypeError" not in result.stdout
  assert "✅ predict: Match" in result.stdout


@patch("ml_switcheroo.testing.harness_generator.get_adapter")
def test_flax_rngs_injection(mock_get_adapter, tmp_path):
  """Verifies the behavior of Flax rngs injection."""
  adapter = SafeFlaxAdapter()
  mock_get_adapter.return_value = adapter
  src_file = tmp_path / "model_nnx_src.py"
  src_file.write_text("def init(x): return x")
  tgt_file = tmp_path / "model_nnx_tgt.py"
  tgt_file.write_text("\ndef init(rngs, x):\n    return x # Echo\n")
  harness_path = tmp_path / "verify_nnx.py"
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="flax_nnx")
  result = _run_harness(harness_path)
  assert "✅ init: Match" in result.stdout
