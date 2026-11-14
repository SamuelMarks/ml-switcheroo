"""Test suite for Rewriter error bubbling and EscapeHatch behavior."""

from __future__ import annotations

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Minimal SemanticsManager double defining mapped and unmapped operations."""

  def __init__(self) -> None:
    """Initialize MockSemantics with 'good' and unmapped 'bad' operations."""
    super().__init__(load_defaults=False)
    self.data = {
      "bad": {"variants": {"jax": None}},
      "good": {"variants": {"jax": {"api": "j.good"}}},
    }
    self.import_data: dict[str, RuntimeTensorOrValue] = {}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Resolve concrete source call names containing 'bad' or 'good'.

    Args:
        name: Identifier or operation name string.

    Returns:
        Matching ``(abstract_id, details)`` tuple or ``None``.
    """
    if "bad" in name:
      return ("bad", self.data["bad"])
    if "good" in name:
      return ("good", self.data["good"])
    return None

  def resolve_variant(self, abstract_id: str, target_framework: str) -> dict[str, JsonValue] | None:
    """Return the variant dictionary for ``(abstract_id, target_framework)`` if present.

    Args:
        abstract_id: Canonical abstract operation identifier.
        target_framework: Target framework key.

    Returns:
        Variant dictionary or ``None``.
    """
    op_entry = self.data.get(abstract_id, {})
    variants = op_entry.get("variants")
    if isinstance(variants, dict):
      var = variants.get(target_framework)
      if isinstance(var, dict):
        return var
    return None

  def is_verified(self, _id: str) -> bool:
    """Mark all operations as verified for bubbling tests.

    Args:
        _id: Abstract operation identifier.

    Returns:
        Always ``True``.
    """
    return True


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a strict-mode TestRewriter from PyTorch to JAX.

  Returns:
      Configured ``TestRewriter`` instance.
  """
  return TestRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True))


def rewrite_stmt(rewriter: TestRewriter, code: str) -> str:
  """Parse and rewrite ``code`` through ``rewriter``.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python source code snippet to transpile.

  Returns:
      Transpiled Python source code string.
  """
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_single_failure_bubbling(rewriter: TestRewriter) -> None:
  """Verify an unmapped operation bubbles failure markers to the enclosing statement.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite_stmt(rewriter, "x = torch.bad(y)")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_nested_failure_bubbling(rewriter: TestRewriter) -> None:
  """Verify nested unmapped calls inside valid calls bubble failure markers cleanly.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite_stmt(rewriter, "x = torch.good(torch.bad(y))")
  assert EscapeHatch.START_MARKER in res
  assert "No mapping" in res


def test_multiple_failures_deduplicated(rewriter: TestRewriter) -> None:
  """Verify duplicate failure messages within the same statement are deduplicated.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite_stmt(rewriter, "l = [torch.bad(1), torch.bad(2)]")
  assert res.count("No mapping") == 1


def test_unknown_strict_mode(rewriter: TestRewriter) -> None:
  """Verify unknown APIs emit EscapeHatch failure markers when strict_mode is enabled.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  res = rewrite_stmt(rewriter, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER in res
  assert "API 'torch.unknown' not found" in res


def test_unknown_lax_mode() -> None:
  """Verify unknown APIs pass through without EscapeHatch markers when strict_mode is disabled."""
  rw = TestRewriter(MockSemantics(), RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False))
  res = rewrite_stmt(rw, "y = torch.unknown(x)")
  assert EscapeHatch.START_MARKER not in res
