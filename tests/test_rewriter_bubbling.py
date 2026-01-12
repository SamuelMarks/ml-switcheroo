"""
Tests for verification failure bubbling via TestRewriter.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock semantics with working and broken operations."""

  def __init__(self):
    self.data = {
      "bad": {"variants": {"jax": None}},
      "good": {"variants": {"jax": {"api": "j.good"}}},
    }
    self.import_data = {}
    self.framework_configs = {}

  def get_definition(self, name):
    """Dynamic lookup."""
    if "bad" in name:
      return "bad", self.data["bad"]
    if "good" in name:
      return "good", self.data["good"]
    return None

  def resolve_variant(self, aid, t):
    """Variant resolution."""
    return self.data.get(aid, {}).get("variants", {}).get(t)

  def is_verified(self, _id):
    """Always verified."""
    return True


@pytest.fixture
def rewriter():
  """Returns strict mode rewriter."""
  return PivotRewriter(
    MockSemantics(),
    RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True),
  )


def rewrite_stmt(rewriter, code):
  """Rewrite helper."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_single_failure_bubbling(rewriter):
  """Verify single call failure marks the line."""
  res = rewrite_stmt(rewriter, "x = torch.bad(y)")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_nested_failure_bubbling(rewriter):
  """Verify nested failure bubbles to top statement."""
  res = rewrite_stmt(rewriter, "x = torch.good(torch.bad(y))")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_multiple_failures_deduplicated(rewriter):
  """Verify duplicate error messages are collapsed."""
  res = rewrite_stmt(rewriter, "l = [torch.bad(1), torch.bad(2)]")
  assert res.count("No mapping") == 1


def test_unknown_strict_mode(rewriter):
  """Verify unknown API failure in strict mode."""
  res = rewrite_stmt(rewriter, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER in res
  assert "API 'torch.unknown' not found" in res


def test_unknown_lax_mode():
  """Verify unknown API pass-through in lax mode."""
  rw = PivotRewriter(
    MockSemantics(),
    RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False),
  )
  res = rewrite_stmt(rw, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER not in res
