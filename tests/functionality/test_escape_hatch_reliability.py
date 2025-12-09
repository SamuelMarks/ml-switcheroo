"""
Tests for Escape Hatch Reliability.

Verifies that:
1. Partial mutations are reverted (Verbatim Preservation).
   - If `torch.abs()` works but `torch.fail()` fails in the same statement,
     the whole statement should remain `torch.abs()...` and NOT `jax.numpy.abs()...`.
2. END markers are correctly added.
3. START markers are correctly added.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """
  Mock Manager configured with:
  - 1 Working Op (good_op)
  - 1 Failing Op (bad_op)
  """

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}  # Required by ASTEngine
    self.framework_configs = {}

    # Good Op: torch.good -> jax.good
    self._inject("good_op", "torch.good", "jax.good")

    # Bad Op: torch.bad -> None (triggers failure)
    self._inject("bad_op", "torch.bad", None)

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    variants = {"torch": {"api": s_api}}
    if t_api:
      variants["jax"] = {"api": t_api}
    else:
      variants["jax"] = None  # Explicit failure

    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  """Parses code, runs rewriter, returns generated source."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_verbatim_preservation_on_partial_failure(rewriter):
  """
  Scenario: A statement contains one translatable call and one failing call.
  Input: res = torch.good(x) + torch.bad(y)

  Behavior:
  - 'torch.good' visits first, rewrites to 'jax.good' (in updated_node).
  - 'torch.bad' visits, fails, triggers _report_failure.
  - leave_SimpleStatementLine sees error.

  Expectation:
  - Result MUST contain 'torch.good' (original), NOT 'jax.good' (partial mutation).
  - Markers must be present.
  """
  code = "res = torch.good(x) + torch.bad(y)"
  result = rewrite_code(rewriter, code)

  # 1. Markers
  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result

  # 2. Verbatim Check (Crucial)
  # The output should have reverted the successful rewrite of good_op
  # because the statement as a whole failed validation.
  assert "torch.good(x)" in result
  assert "jax.good(x)" not in result
  assert "torch.bad(y)" in result


def test_end_marker_presence(rewriter):
  """
  Verify the END marker and the Ellipsis sentinel are emitted.
  """
  code = "y = torch.bad(x)"
  result = rewrite_code(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result
  # Check for the ellipsis (...)
  assert "..." in result


def test_nested_call_failure_bubbling(rewriter):
  """
  Scenario: Failure deep inside a call.
  Input: y = torch.good(torch.bad(x))
  """
  code = "y = torch.good(torch.bad(x))"
  result = rewrite_code(rewriter, code)

  # Should preserve the outer call as torch.good
  assert "torch.good" in result
  assert "jax.good" not in result
  assert EscapeHatch.START_MARKER in result


def test_multiple_statements_handled_independently(rewriter):
  """
  Verify that a failure in one line doesn't affect the next line.
  """
  code = """ 
y = torch.bad(x) 
z = torch.good(x) 
"""
  result = rewrite_code(rewriter, code)

  print(result)

  # Line 1: Failed -> Wrapped & Verbatim
  assert "torch.bad" in result
  assert result.count(EscapeHatch.START_MARKER) == 1

  # Line 2: Succeeded -> Transformed & No Markers
  assert "jax.good(x)" in result
  assert "torch.good" not in result
