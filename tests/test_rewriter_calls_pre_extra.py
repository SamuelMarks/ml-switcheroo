"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks_registry import register_hook
import pytest
from typing import Tuple, Generator


class DummyRewriter:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.context: MagicMock = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics: MagicMock = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w: str) -> None:
    pass


@pytest.fixture(autouse=True)
def _cleanup() -> Generator[None, None, None]:
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change() -> None:
  """Test element."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    return node

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change() -> None:
  """Test element."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    return getattr(cst.parse_statement("b = 1"), "body")[0]

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
