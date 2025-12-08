"""
Tests for Smart Import Injection.

Verifies that:
1. Imports are inserted AFTER docstrings and `__future__` imports.
2. Standard aliases (`jnp`, `tf`, `mx`) are injected only when used.
3. Target root imports (`import jax`, `import tensorflow`) are injected only when required.
4. Submodule injection via data-driven maps works correctly.
"""

import libcst as cst
from ml_switcheroo.core.import_fixer import ImportFixer

# Dummy map for initialization
TEST_MAP = {
  "torch.nn": ("flax", "linen", "nn"),
  "torch.optim": ("optax", None, "optim"),
}

# Mimic the defaults provided by SemanticsManager
DEFAULT_ALIASES = {
  "jax": ("jax.numpy", "jnp"),
  "tensorflow": ("tensorflow", "tf"),
  "mlx": ("mlx.core", "mx"),
  "numpy": ("numpy", "np"),
}


def apply_fixer(code: str, target="jax", d=None) -> str:
  """Run ImportFixer with default settings."""
  tree = cst.parse_module(code)
  m = d if d is not None else TEST_MAP
  fixer = ImportFixer("torch", target, submodule_map=m, alias_map=DEFAULT_ALIASES)
  new_tree = tree.visit(fixer)
  return new_tree.code


def test_insert_after_docstring():
  """
  Input:
      "Docstring"
      x = jax.numpy.abs(1)
  Expect: jax imports appear after the string.
  """
  code = '"""My Module Doc."""\nx = jax.numpy.abs(1)'
  result = apply_fixer(code)

  lines = result.splitlines()
  assert '"""My Module Doc."""' in lines[0]
  assert "import jax" in result

  # Ensure import is NOT before docstring (docstring must be index 0)
  assert result.find("import jax") > result.find("Doc")


def test_insert_after_future():
  """
  Input:
      from __future__ import annotations
      x = jax.numpy.abs(1)
  Expect: Imports after future.
  """
  code = "from __future__ import annotations\nx = jax.numpy.abs(1)"
  result = apply_fixer(code)

  assert "from __future__" in result
  assert "import jax" in result

  # Future must strictly be first
  assert result.find("from __future__") < result.find("import jax")


def test_insert_at_top_if_no_preamble():
  """
  Input:
      x = jax.numpy.abs(1)
  Expect: Imports first.
  """
  code = "x = jax.numpy.abs(1)"
  result = apply_fixer(code)

  assert result.startswith("import jax")


def test_smart_injection_jnp_usage():
  """
  Scenario: Transpiled code uses `jnp.array(...)`.
  Expect: `import jax.numpy as jnp` is injected.
  Expect: `import jax` is NOT injected (unless `jax.` is also used).
  """
  code = "x = jnp.array([1, 2])"
  result = apply_fixer(code, target="jax")

  assert "import jax.numpy as jnp" in result
  assert "import jax\n" not in result


def test_smart_injection_jax_usage():
  """
  Scenario: Transpiled code uses `jax.lax.scan(...)`.
  Expect: `import jax` is injected.
  Expect: `jnp` import is NOT injected.
  """
  code = "y = jax.lax.scan(f, x, xs)"
  result = apply_fixer(code, target="jax")

  assert "import jax" in result
  assert "as jnp" not in result


def test_smart_injection_tensorflow():
  """
  Scenario: Transpiled code uses `tf.math.add(...)`.
  Target: tensorflow.
  Expect: `import tensorflow as tf` injection.
  """
  code = "y = tf.math.add(x, x)"
  result = apply_fixer(code, target="tensorflow")

  # Should detect use of 'tf' alias via FRAMEWORK_ALIAS_MAP
  assert "import tensorflow as tf" in result


def test_smart_injection_mlx():
  """
  Scenario: Transpiled code uses `mx.array(...)`.
  Target: mlx.
  Expect: `import mlx.core as mx` injection.
  """
  code = "arr = mx.array([1, 2, 3])"
  result = apply_fixer(code, target="mlx")

  # Should detect use of 'mx' alias
  assert "import mlx.core as mx" in result


def test_smart_injection_numpy():
  """
  Scenario: Transpiled code uses `np.array(...)`.
  Target: numpy.
  Expect: `import numpy as np` injection.
  """
  code = "arr = np.array([1, 2, 3])"
  result = apply_fixer(code, target="numpy")

  assert "import numpy as np" in result


def test_smart_injection_no_usage():
  """
  Scenario: Transpiled code consists of standard python.
  Expect: No injections at all.
  """
  code = "print('Hello World')"
  result = apply_fixer(code)

  assert "import jax" not in result
  assert "as jnp" not in result


def test_no_double_injection_if_exists():
  """
  Scenario: Source already had `import jax.numpy as jnp` (preserved).
  Expect: Do NOT inject a second `import jax.numpy as jnp`.
  """
  code = "import jax.numpy as jnp\nx = jnp.ones(3)"
  result = apply_fixer(code)

  # We expect exactly one occurrence of import line
  assert result.count("import jax.numpy as jnp") == 1


def test_smart_injection_submodule_data_driven():
  """
  Scenario: User code uses `nn.Dense`. `nn` undefined.
  Map: ("flax", "linen", "nn") -> Submodule import.
  Expect: `from flax import linen as nn`.
  """
  map_data = {"torch.nn": ("flax", "linen", "nn")}
  code = "layer = nn.Dense(10)"  # 'nn' is used but undefined

  result = apply_fixer(code, d=map_data)

  assert "from flax import linen as nn" in result
  assert "import flax" not in result  # Should be specific


def test_smart_injection_root_import():
  """
  Scenario: User code uses `optax`.
  Map: ("optax", None, "optax"). `None` sub indicates root import.
  Expect: `import optax`.
  """
  map_data = {"torch.optim": ("optax", None, "optax")}
  # User calls optax.adam()
  code = "opt = optax.adam(0.1)"

  result = apply_fixer(code, d=map_data)

  # Should generate 'import optax'
  assert "import optax" in result
  # Should NOT generate 'from optax import ...'
  assert "from optax" not in result
