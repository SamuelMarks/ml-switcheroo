"""Test module."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


def setup_ctx():
  """Test function."""
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  return RewriterContext(semantics=SemanticsManager(), config=config)


def test_auxiliary_leave_decorator_rename_call():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Call(func=cst.Name("test")))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same2(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same2(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert res is loop
