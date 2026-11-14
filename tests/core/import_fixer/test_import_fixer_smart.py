"""Test suite for the Import Fixer Smart module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan
from ml_switcheroo.semantics.manager import SemanticsManager


def solve_and_fix(code: str, target_framework: str = "jax", alias_map: dict[str, tuple[str, str]] | None = None) -> str:
  """Execute `ImportResolver` to support solve and fix in import fixer smart tests.

  Args:
      code: Python or DSL source code string to parse or transpile.
      target_framework: Target framework identifier string.
      alias_map: Input `alias_map` parameter value (`dict[str, tuple[str, str]] | None`).

  Returns:
      Resulting `str` from solve and fix.
  """
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
  plan: ResolutionPlan = resolver.resolve(tree, target_framework)

  from ml_switcheroo.core.scanners import GlobalUsageScanner

  scanner = GlobalUsageScanner()
  tree.visit(scanner)

  fixer = ImportFixer(plan=plan, source_frameworks={"torch"}, used_names=scanner.used_names)
  new_tree = tree.visit(fixer)
  return new_tree.code


def test_smart_injection_jnp_usage() -> None:
  """Verify `solve_and_fix` emits `import jax.numpy as jnp` for smart injection jnp usage."""
  code: str = "x = jnp.array([1])"
  result: str = solve_and_fix(code, "jax")
  assert "import jax.numpy as jnp" in result
  assert "import jax\n" not in result


def test_smart_injection_tensorflow() -> None:
  """Verify `solve_and_fix` emits `import tensorflow as tf` for smart injection tensorflow."""
  code: str = "y = tf.math.add(x, x)"
  result: str = solve_and_fix(code, "tensorflow")
  assert "import tensorflow as tf" in result


def test_no_double_injection() -> None:
  """Verify `solve_and_fix` returns `1` for no double injection."""
  code: str = "import jax.numpy as jnp\nx = jnp.ones(3)"
  result: str = solve_and_fix(code, "jax")
  assert result.count("import jax.numpy as jnp") == 1
