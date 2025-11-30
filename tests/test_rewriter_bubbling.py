"""
Tests for PivotRewriter Error Bubbling.

Verifies that rewrite failures in nested expressions (like Call nodes)
are correctly bubbled up to the statement line so EscapeHatch markers
can be attached to the valid AST node.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  def __init__(self):
    # Skip init to avoid file load, manually setup logic
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}

    # 1. Known Good
    self._inject("good_op", "torch", "torch.good", "jax", "jax.good")

    # 2. Explicit Gap (Tier C -> None)
    self._inject("bad_op", "torch", "torch.bad", "jax", None)

  def _inject(self, name, s_fw, s_api, t_fw, t_api):
    variants = {s_fw: {"api": s_api}}
    if t_api is None:
      variants[t_fw] = None
    else:
      variants[t_fw] = {"api": t_api}

    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockSemantics()
  # Use config object for initialization
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite_stmt(rewriter, code):
  """Parses code, applies rewriter, and returns generated code."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_single_failure_bubbling(rewriter):
  """
  Scenario: Single bad call in a statement.
  Input: x = torch.bad(y)
  Expect: Comment on statement line.
  """
  code = "x = torch.bad(y)"
  result = rewrite_stmt(rewriter, code)

  # Assert markers are present
  assert EscapeHatch.START_MARKER in result
  assert "No mapping defined" in result
  # Code preserved
  assert "x = torch.bad(y)" in result


def test_nested_failure_bubbling(rewriter):
  """
  Scenario: Failure inside nested call argument.
  Input: x = torch.good(torch.bad(y))
  Expect: Warning on the line 'x = ...'.
  """
  code = "x = torch.good(torch.bad(y))"
  result = rewrite_stmt(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  # We expect 'good' to translate (if arguments were valid), but bubbling
  # means the whole line is marked as having issues.
  assert "Call 'torch.bad' not translated" not in result  # Logic specific text
  assert "No mapping defined" in result


def test_multiple_failures_deduplicated(rewriter):
  """
  Scenario: Multiple failures in one line.
  Input: x = torch.bad(a) + torch.bad(b)
  Expect: Markers attached, message contains error once or joined.
  """
  code = "res = [torch.bad(1), torch.bad(2)]"
  result = rewrite_stmt(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  # Check deduplication helper logic
  # We count how many times the error msg sub-string appears in the comment block.
  # It should effectively be listed, but duplicates removed by set() in code.
  assert result.count("No mapping defined") == 1


def test_unknown_strict_mode(rewriter):
  """
  Scenario: Strict mode on, unknown function called.
  """
  assert rewriter.strict_mode is True
  code = "y = torch.unknown_func(x)"
  result = rewrite_stmt(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  assert "API 'torch.unknown_func' not found" in result


def test_unknown_lax_mode():
  """
  Scenario: Strict mode off, unknown function is ignored (no bubbling).
  """
  semantics = MockSemantics()

  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  lax_rewriter = PivotRewriter(semantics, config)

  code = "y = torch.unknown_func(x)"
  result = rewrite_stmt(lax_rewriter, code)

  assert EscapeHatch.START_MARKER not in result
  assert "torch.unknown_func" in result
