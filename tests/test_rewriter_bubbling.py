import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {"bad": {"variants": {"jax": None}}, "good": {"variants": {"jax": {"api": "j.good"}}}}
    self.import_data = {}
    self.framework_configs = {}

  def get_definition(self, name):
    if "bad" in name:
      return "bad", self.data["bad"]
    if "good" in name:
      return "good", self.data["good"]
    return None

  def resolve_variant(self, aid, t):
    return self.data.get(aid, {}).get("variants", {}).get(t)

  def is_verified(self, _id):
    return True


@pytest.fixture
def rewriter():
  return PivotRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True))


def rewrite_stmt(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


def test_single_failure_bubbling(rewriter):
  res = rewrite_stmt(rewriter, "x = torch.bad(y)")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_nested_failure_bubbling(rewriter):
  res = rewrite_stmt(rewriter, "x = torch.good(torch.bad(y))")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_multiple_failures_deduplicated(rewriter):
  res = rewrite_stmt(rewriter, "l = [torch.bad(1), torch.bad(2)]")
  assert res.count("No mapping") == 1


def test_unknown_strict_mode(rewriter):
  res = rewrite_stmt(rewriter, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER in res
  assert "API 'torch.unknown' not found" in res


def test_unknown_lax_mode():
  rw = PivotRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False))
  res = rewrite_stmt(rw, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER not in res
