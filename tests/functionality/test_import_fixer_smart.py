"""Test suite for the Import Fixer Smart module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.semantics.manager import SemanticsManager


def solve_and_fix(code, target_fw="jax", alias_map=None):
  """Helper to solve and fix."""
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_framework_aliases.return_value = alias_map or {
    "jax": ("jax.numpy", "jnp"),
    "tensorflow": ("tensorflow", "tf"),
    "mlx": ("mlx.core", "mx"),
    "numpy": ("numpy", "np"),
  }
  mgr.get_import_map.return_value = {}
  resolver = ImportResolver(mgr)
  tree = cst.parse_module(code)
  plan = resolver.resolve(tree, target_fw)
  fixer = ImportFixer(plan=plan, source_fws={"torch"})
  return tree.visit(fixer).code


def test_smart_injection_jnp_usage():
  """Verifies the behavior of smart injection jnp usage."""
  code = "x = jnp.array([1])"
  result = solve_and_fix(code, "jax")
  assert "import jax.numpy as jnp" in result
  assert "import jax\n" not in result


def test_smart_injection_tensorflow():
  """Verifies the behavior of smart injection TensorFlow."""
  code = "y = tf.math.add(x, x)"
  result = solve_and_fix(code, "tensorflow")
  assert "import tensorflow as tf" in result


def test_no_double_injection():
  """Verifies the behavior of no double injection."""
  code = "import jax.numpy as jnp\nx = jnp.ones(3)"
  result = solve_and_fix(code, "jax")
  assert result.count("import jax.numpy as jnp") == 1
