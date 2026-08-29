"""Test suite for the Rewriter Bubbling module."""

from typing import Any, Dict, Optional, Tuple

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: Dict[str, Any] = {"bad": {"variants": {"jax": None}}, "good": {"variants": {"jax": {"api": "j.good"}}}}
    self.import_data: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}

  def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Mock implementation of get definition."""
    if "bad" in name:
      return ("bad", self.data["bad"])
    if "good" in name:
      return ("good", self.data["good"])
    return None

  def resolve_variant(self, aid: str, t: str) -> Optional[Dict[str, Any]]:
    """Mock implementation of resolve variant."""
    return self.data.get(aid, {}).get("variants", {}).get(t)

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified."""
    return True


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  return PivotRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True))


def rewrite_stmt(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites stmt."""
  tree: cst.Module = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_single_failure_bubbling(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of single successfully handling failure bubbling."""
  res: str = rewrite_stmt(rewriter, "x = torch.bad(y)")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_nested_failure_bubbling(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of nested successfully handling failure bubbling."""
  res: str = rewrite_stmt(rewriter, "x = torch.good(torch.bad(y))")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_multiple_failures_deduplicated(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of multiple failures deduplicated."""
  res: str = rewrite_stmt(rewriter, "l = [torch.bad(1), torch.bad(2)]")
  assert res.count("No mapping") == 1


def test_unknown_strict_mode(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of unknown strict mode."""
  res: str = rewrite_stmt(rewriter, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER in res
  assert "API 'torch.unknown' not found" in res


def test_unknown_lax_mode() -> None:
  """Verifies the behavior of unknown lax mode."""
  rw: PivotRewriter = PivotRewriter(
    MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  )
  res: str = rewrite_stmt(rw, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER not in res
