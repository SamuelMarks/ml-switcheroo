"""
Tests for Smart Import Injection via ResolutionPlan.

Verifies that:
1. Imports are inserted based on resolution.
2. Standard aliases are injected only when used.
3. Target root imports are injected only when required.
4. Submodule injection via mappings works correctly.
"""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.semantics.manager import SemanticsManager


def solve_and_fix(code, target_fw="jax", alias_map=None):
  # Mock Semantics
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_framework_aliases.return_value = alias_map or {
    "jax": ("jax.numpy", "jnp"),
    "tensorflow": ("tensorflow", "tf"),
    "mlx": ("mlx.core", "mx"),
    "numpy": ("numpy", "np"),
  }
  # Import map mock
  mgr.get_import_map.return_value = {}

  resolver = ImportResolver(mgr)
  tree = cst.parse_module(code)
  plan = resolver.resolve(tree, target_fw)

  fixer = ImportFixer(plan=plan, source_fws={"torch"})
  return tree.visit(fixer).code


def test_smart_injection_jnp_usage():
  """
  Scenario: Transpiled code uses `jnp.array(...)`.
  Expect: `import jax.numpy as jnp` is injected.
  """
  code = "x = jnp.array([1])"
  result = solve_and_fix(code, "jax")
  assert "import jax.numpy as jnp" in result
  assert "import jax\n" not in result


def test_smart_injection_tensorflow():
  """
  Scenario: Transpiled code uses `tf.math.add(...)`.
  Target: tensorflow.
  Expect: `import tensorflow as tf` injection.
  """
  code = "y = tf.math.add(x, x)"
  result = solve_and_fix(code, "tensorflow")
  assert "import tensorflow as tf" in result


def test_no_double_injection():
  """
  Scenario: Source already had `import jax.numpy as jnp`.
  Expect: Do NOT inject a second `import jax.numpy as jnp`.
  """
  code = "import jax.numpy as jnp\nx = jnp.ones(3)"
  result = solve_and_fix(code, "jax")

  # We expect exactly one occurrence of import line
  assert result.count("import jax.numpy as jnp") == 1
