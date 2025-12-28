"""
Tests for Strict Mode Shape Guards (Transpiler Feature).

Note: This file previously tested the Fuzzer's output_shape_calc Logic.
It has been repurposed and expanded to verify the Transpiler's runtime
guard injection feature, satisfying the requirement to test "Strict Shape Guards".
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.core.hooks import HookContext

# --- Mock Infrastructure ---


class MockShapeSemantics(SemanticsManager):
  def __init__(self):
    # Bypass init
    self.data = {}
    self._reverse_index = {}
    self.framework_configs = {}
    self._key_origins = {}
    self.import_data = {}
    self._known_rng_methods = set()

    # Define Conv2d with Rank constraint (4)
    self.data["Conv2d"] = {
      "std_args": [{"name": "input", "rank": 4}, {"name": "weight", "rank": 4}],
      "variants": {
        "torch": {"api": "torch.nn.functional.conv2d", "args": {"input": "input"}},
        "jax": {"api": "jax.lax.conv", "args": {"input": "lhs", "weight": "rhs"}},
      },
    }
    self._reverse_index["torch.nn.functional.conv2d"] = ("Conv2d", self.data["Conv2d"])

    # Define Linear (No constraint)
    self.data["Linear"] = {
      "std_args": ["x", "w"],
      "variants": {"torch": {"api": "torch.nn.functional.linear"}, "jax": {"api": "jax.nn.linear"}},
    }
    self._reverse_index["torch.nn.functional.linear"] = ("Linear", self.data["Linear"])

    # Pre-populate index for intermediate modules to handle strict mode checks on attributes
    # The Mock needs to return valid mappings for these so BaseRewriter attribute check doesn't fail
    self.data["torch_nn"] = {"variants": {"jax": {"api": "jax.nn"}}}
    self._reverse_index["torch.nn"] = ("torch_nn", self.data["torch_nn"])

    self.data["torch_nn_functional"] = {"variants": {"jax": {"api": "jax.nn"}}}
    self._reverse_index["torch.nn.functional"] = ("torch_nn_functional", self.data["torch_nn_functional"])

  def get_definition(self, name):
    # Return empty dict for intermediate modules to pass attribute check
    if name in self._reverse_index:
      return self._reverse_index[name]
    return None

  def get_framework_config(self, fw):
    return {}


@pytest.fixture
def rewriter_factory():
  semantics = MockShapeSemantics()

  def create(strict=False):
    config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=strict)
    return PivotRewriter(semantics, config)

  return create


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  # If preamble exists, inject it manually for source str generation logic in test
  # BaseRewriter.leave_Module usually handles this logic in real execution
  if hasattr(rewriter, "_module_preamble") and rewriter._module_preamble:
    preamble_stmts = []
    for code_str in rewriter._module_preamble:
      preamble_stmts.extend(cst.parse_module(code_str).body)
    new_tree = new_tree.with_changes(body=preamble_stmts + list(new_tree.body))

  return new_tree.code


# --- Tests ---


def test_strict_guard_injection(rewriter_factory):
  """
  Scenario: Op with Rank=4 constraint. Strict Mode ENABLED.
  Expect: _check_rank wrapper around argument.
  """
  rewriter = rewriter_factory(strict=True)
  code = "y = torch.nn.functional.conv2d(input=x, weight=w)"

  res = rewrite(rewriter, code)

  # 1. Check helper injection via preamble logic
  assert "def _check_rank(x, rank):" in res

  # 2. Check wrappers
  # Target args map: input -> lhs, weight -> rhs
  # Wrapper should be applied to value 'x' and 'w'
  assert "_check_rank(x, 4)" in res
  assert "_check_rank(w, 4)" in res

  # 3. Check target API
  assert "jax.lax.conv(lhs=_check_rank(" in res.replace(" ", "")


def test_lax_mode_no_injection(rewriter_factory):
  """
  Scenario: Op with Rank=4 constraint. Strict Mode DISABLED.
  Expect: No wrapper.
  """
  rewriter = rewriter_factory(strict=False)
  code = "y = torch.nn.functional.conv2d(input=x, weight=w)"

  res = rewrite(rewriter, code)

  assert "_check_rank" not in res
  assert "jax.lax.conv" in res


def test_guard_ignore_no_constraint(rewriter_factory):
  """
  Scenario: Op with no rank constraints (Linear). Strict Mode ENABLED.
  Expect: No wrapper.
  """
  rewriter = rewriter_factory(strict=True)
  code = "y = torch.nn.functional.linear(x, w)"

  res = rewrite(rewriter, code)

  assert "_check_rank" not in res
  assert "jax.nn.linear" in res
