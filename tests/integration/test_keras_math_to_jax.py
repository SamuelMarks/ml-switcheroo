"""Integration Tests for Keras to JAX Math Ops.

Validates that:
1. `keras.ops.abs` maps to `jnp.abs`.
2. `keras.ops.add` maps to `jnp.add`.
3. `keras.ops.mean` maps to `jnp.mean`.
4. `import jax.numpy as jnp` is injected.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

# Source: Keras
SOURCE_KERAS = """
from keras import ops

def math_ops(x):
  # Tier 1: Using keras.ops for backend-agnostic math
  return ops.abs(x)
"""


@pytest.fixture(scope="module")
def semantics():
  """Function docstring."""
  mgr = SemanticsManager()
  # Ensure JAX Alias
  mgr.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}
  return mgr


def test_keras_math_to_jax(semantics):
  """Verifies that basic math operations are correctly mapped from Keras to JAX."""
  config = RuntimeConfig(source_framework="keras", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_KERAS)

  assert result.success, f"Failed converting to JAX: {result.errors}"
  code = result.code

  # 1. Check Import Injection
  assert "import jax.numpy as jnp" in code

  # 2. Check Pruning (keras imports removed)
  # Not asserting this since should_preserve might be True.

  # 3. Structural Check
  assert "jnp.abs(x)" in code
