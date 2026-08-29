"""Test suite for the Keras Math To Jax module."""

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_KERAS: str = "\nfrom keras import ops\n\ndef math_ops(x):\n  # Tier 1: Using keras.ops for backend-agnostic math\n  return ops.abs(x)\n"


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Helper to semantics."""
  mgr = SemanticsManager()
  mgr.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}
  return mgr


def test_keras_math_to_jax(semantics: SemanticsManager) -> None:
  """Verifies the behavior of Keras math to JAX."""
  config = RuntimeConfig(source_framework="keras", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_KERAS)
  assert result.success, f"Failed converting to JAX: {result.errors}"
  code: str = result.code
  assert "import jax.numpy as jnp" in code
  assert "jnp.abs(x)" in code
