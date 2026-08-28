"""Test suite for the Einsum module."""

import pytest
import typing
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.einsum import normalize_einsum


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["einsum_normalizer"] = normalize_einsum
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  einsum_def: dict[str, typing.Any] = {
    "variants": {"jax": {"api": "jax.numpy.einsum", "requires_plugin": "einsum_normalizer"}}
  }
  mgr.get_definition.return_value = ("einsum", einsum_def)
  mgr.get_known_apis.return_value = {"einsum": einsum_def}
  mgr.resolve_variant.side_effect = lambda aid, fw: typing.cast(
    typing.Optional[dict[str, typing.Any]], einsum_def["variants"].get(fw)
  )
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_standard_order_unchanged(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of standard order unchanged."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum("ii", x)')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_swap_operand_and_equation(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of swap operand and equation."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii")')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_multiple_operands_swap(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of multiple operands swap."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum(a, b, "i,j->ij")')
  assert '("i,j->ij", a, b)' in res


def test_interleaved_operands_unsupported_heuristic(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of interleaved operands unsupported heuristic."""
  res: str = rewrite_code(rewriter, "torch.einsum(a, [0], b, [0])")
  assert "jax.numpy.einsum" in res
  assert "(a, [0], b, [0])" in res


def test_variable_equation_ignored(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of variable equation ignored."""
  res: str = rewrite_code(rewriter, "torch.einsum(x, eq)")
  assert "(x, eq)" in res


def test_empty_args(rewriter: PivotRewriter) -> None:
  """Verifies behavior when called with no args."""
  res: str = rewrite_code(rewriter, "y = torch.einsum()")
  assert "jax.numpy.einsum()" in res


def test_equation_only(rewriter: PivotRewriter) -> None:
  """Verifies behavior with only equation string."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum("ii")')
  assert 'jax.numpy.einsum("ii")' in res


def test_trailing_comma(rewriter: PivotRewriter) -> None:
  """Verifies behavior with trailing comma."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii",)')
  assert 'jax.numpy.einsum("ii", x)' in res


def test_equation_in_middle(rewriter: PivotRewriter) -> None:
  """Verifies behavior when equation is in the middle."""
  res: str = rewrite_code(rewriter, 'y = torch.einsum(x, "ii", z)')
  assert 'jax.numpy.einsum("ii", x, z)' in res
