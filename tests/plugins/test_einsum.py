import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.einsum import normalize_einsum


def rewrite_code(rewriter, code):
  """Executes pipeline."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["einsum_normalizer"] = normalize_einsum
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  einsum_def = {
    "variants": {
      "jax": {
        "api": "jax.numpy.einsum",
        "requires_plugin": "einsum_normalizer",
      }
    }
  }

  mgr.get_definition.return_value = ("einsum", einsum_def)
  mgr.get_known_apis.return_value = {"einsum": einsum_def}
  mgr.resolve_variant.side_effect = lambda aid, fw: einsum_def["variants"].get(fw)

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_standard_order_unchanged(rewriter):
  res = rewrite_code(rewriter, 'y = torch.einsum("ii", x)')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_swap_operand_and_equation(rewriter):
  res = rewrite_code(rewriter, 'y = torch.einsum(x, "ii")')
  assert "jax.numpy.einsum" in res
  assert '("ii", x)' in res


def test_multiple_operands_swap(rewriter):
  res = rewrite_code(rewriter, 'y = torch.einsum(a, b, "i,j->ij")')
  assert '("i,j->ij", a, b)' in res


def test_interleaved_operands_unsupported_heuristic(rewriter):
  res = rewrite_code(rewriter, "torch.einsum(a, [0], b, [0])")
  assert "jax.numpy.einsum" in res
  assert "(a, [0], b, [0])" in res


def test_variable_equation_ignored(rewriter):
  res = rewrite_code(rewriter, "torch.einsum(x, eq)")
  assert "(x, eq)" in res
