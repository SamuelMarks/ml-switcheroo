"""
Integration Tests for Keras to JAX Math Ops.

Validates that:
1. `keras.ops.abs` maps to `jnp.abs`.
2. `keras.ops.add` maps to `jnp.add`.
3. `keras.ops.mean` maps to `jnp.mean`.
4. `import jax.numpy as jnp` is injected.
"""

import ast
import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.utils.ast_utils import cmp_ast

# Source: Keras
SOURCE_KERAS = """
import keras
from keras import ops

def math_ops(x, y):
  # Tier 1: Using keras.ops for backend-agnostic math
  a = ops.abs(x)
  b = ops.add(a, y)
  return ops.mean(b)
"""

# Expected: JAX
# Note: ImportFixer automatically injects 'import jax.numpy as jnp' because 'jnp' is used.
EXPECTED_JAX = """
import jax.numpy as jnp

def math_ops(x, y):
  # Tier 1: Using keras.ops for backend-agnostic math
  a = jnp.abs(x)
  b = jnp.add(a, y)
  return jnp.mean(b)
"""


@pytest.fixture(scope="module")
def semantics():
  return SemanticsManager()


def test_keras_math_to_jax(semantics):
  """
  Verifies that basic math operations are correctly mapped from Keras to JAX.
  """
  config = RuntimeConfig(source_framework="keras", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_KERAS)

  assert result.success, f"Failed converting to JAX: {result.errors}"

  code = result.code

  # 1. Check Import Injection
  assert "import jax.numpy as jnp" in code

  # 2. Check Pruning (keras imports removed)
  # The ImportFixer removes 'import keras' and 'from keras import ops'
  # because they are from source 'keras' and target is 'jax'.
  assert "import keras" not in code
  assert "from keras import ops" not in code

  # 3. Check AST Logic
  # Normalize whitespaces/newlines for robust comparison
  generated_ast = ast.parse(code)
  expected_ast = ast.parse(EXPECTED_JAX)

  if not cmp_ast(generated_ast, expected_ast):
    print(f"\n--- Expected ---\n{EXPECTED_JAX}")
    print(f"\n--- Actual ---\n{result.code}")
    pytest.fail("AST Mismatch")
