"""Test suite for the Einsum module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.einsum import normalize_einsum
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for einsum tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("einsum_normalizer")(normalize_einsum)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  einsum_def = {"variants": {"jax": {"api": "jax.numpy.einsum", "requires_plugin": "einsum_normalizer"}}}
  mgr.get_definition.return_value = ("einsum", einsum_def)
  mgr.get_known_apis.return_value = {"einsum": einsum_def}
  mgr.resolve_variant.side_effect = lambda aid, fw: einsum_def["variants"].get(fw)
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_standard_order_unchanged(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.einsum` for standard order unchanged.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum("ii", x)')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_swap_operand_and_equation(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.einsum` for swap operand and equation.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii")')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_multiple_operands_swap(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles multiple operands swap as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum(a, b, "i,j->ij")')
  assert '("i,j->ij", a, b)' in res


def test_interleaved_operands_unsupported_heuristic(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles interleaved operands unsupported heuristic as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, "torch.einsum(a, [0], b, [0])")
  assert "jax.numpy.einsum" in res
  assert "(a, [0], b, [0])" in res


def test_variable_equation_ignored(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `(x, eq)` for variable equation ignored.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, "torch.einsum(x, eq)")
  assert "(x, eq)" in res


def test_empty_args(rewriter: PivotRewriter) -> None:
  """Verifies behavior when called with no args.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, "y = torch.einsum()")
  assert "jax.numpy.einsum()" in res


def test_equation_only(rewriter: PivotRewriter) -> None:
  """Verifies behavior with only equation string.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum("ii")')
  assert 'jax.numpy.einsum("ii")' in res


def test_trailing_comma(rewriter: PivotRewriter) -> None:
  """Verifies behavior with trailing comma.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii",)')
  assert 'jax.numpy.einsum("ii", x)' in res


def test_equation_in_middle(rewriter: PivotRewriter) -> None:
  """Verifies behavior when equation is in the middle.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii", z)')
  assert 'jax.numpy.einsum("ii", x, z)' in res
