"""
Tests for Einsum Normalizer Plugin.

Verifies that `einsum` arguments are standardized such that the equation
string is always the first argument.

Scenarios:
1. `einsum("ii", x)` -> Unchanged (already correct).
2. `einsum(x, "ii")` -> `einsum("ii", x)` (Equation moved to front).
3. `einsum(x, y, "ij,jk->ik")` -> `einsum("ij,jk->ik", x, y)`.
4. Mixed calls with multiple variables are handled.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.einsum import normalize_einsum


# Helper to executing rewrite on a code snippet
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook & ensure plugins loaded state is set
  hooks._HOOKS["einsum_normalizer"] = normalize_einsum
  hooks._PLUGINS_LOADED = True

  # 2. Setup Mock Semantics
  mgr = MagicMock()

  # We map torch.einsum -> jax.numpy.einsum (via normalizer)
  einsum_def = {
    "requires_plugin": "einsum_normalizer",
    # std_args is somewhat flexible for variadic, typically ["equation", "operands"]
    "std_args": ["equation", "operands"],
    "variants": {
      "torch": {"api": "torch.einsum"},
      "jax": {"api": "jax.numpy.einsum", "requires_plugin": "einsum_normalizer"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.einsum":
      return "einsum", einsum_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"einsum": einsum_def}
  mgr.is_verified.return_value = True

  # 3. Config for Torch -> JAX
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_standard_order_unchanged(rewriter):
  """
  Scenario: `torch.einsum("ii->i", x)`
  Already normalized.
  """
  code = 'y = torch.einsum("ii->i", x)'
  result = rewrite_code(rewriter, code)

  # Should be converted to jax.numpy.einsum via standard rename (handled by plugin)
  # Arguments should be untouched by plugin.
  assert "jax.numpy.einsum" in result
  clean = result.replace(" ", "")
  assert '("ii->i",x)' in clean


def test_swap_operand_and_equation(rewriter):
  """
  Scenario: `torch.einsum(x, "ii->i")`
  Should become `jax.numpy.einsum("ii->i", x)`
  """
  code = 'y = torch.einsum(x, "ii->i")'
  result = rewrite_code(rewriter, code)

  assert "jax.numpy.einsum" in result
  clean = result.replace(" ", "")
  # Check equation is first
  assert '("ii->i",x)' in clean


def test_multiple_operands_swap(rewriter):
  """
  Scenario: `torch.einsum(a, b, "i,j->ij")`
  Should become `jax.numpy.einsum("i,j->ij", a, b)`
  """
  code = 'y = torch.einsum(a, b, "i,j->ij")'
  result = rewrite_code(rewriter, code)

  clean = result.replace(" ", "")
  assert '("i,j->ij",a,b)' in clean


def test_interleaved_operands_unsupported_heuristic(rewriter):
  """
  Scenario: `torch.einsum(a, [0, 1], b, [1, 2], [0, 2])` (Implicit mode).
  This does NOT involve a string equation.
  The normalizer looks for a string. If absent, it should leave it alone
  (JAX/NumPy supports standard implicit form too).
  """
  code = "y = torch.einsum(a, [0], b, [0])"
  result = rewrite_code(rewriter, code)

  # Should just rename function, no args shuffle
  assert "jax.numpy.einsum" in result
  # Order preserved
  # Note: stripping spaces
  clean = result.replace(" ", "")
  assert "(a,[0],b,[0])" in clean


def test_variable_equation_ignored(rewriter):
  """
  Scenario: `eq = '...'; torch.einsum(x, eq)`
  Since variable 'eq' is not a String Literal in AST, we cannot detect it safely.
  Expect pass-through (no rotation).
  """
  code = "y = torch.einsum(x, eq)"
  result = rewrite_code(rewriter, code)

  clean = result.replace(" ", "")
  # Order preserved (plugin returns unmodified)
  assert "(x,eq)" in clean
