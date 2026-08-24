"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks_registry import register_hook
import pytest


class DummyRewriter:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.context = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w):
    pass


@pytest.fixture(autouse=True)
def _cleanup():
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change():
  """Test element."""
  rewriter = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node, ctx):
    return node

  original = cst.parse_expression("foo_()")
  res = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change():
  """Test element."""
  rewriter = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node, ctx):
    return cst.parse_statement("b = 1").body[0]

  original = cst.parse_expression("foo_()")
  res = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
