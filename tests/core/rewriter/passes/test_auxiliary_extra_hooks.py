"""Test module."""

import libcst as cst
import typing
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


def setup_ctx() -> RewriterContext:
  """Test function."""
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  return RewriterContext(semantics=SemanticsManager(), config=config)


def test_auxiliary_visit_simplestatementline() -> None:
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  res: typing.Any = p.visit_SimpleStatementLine(cst.SimpleStatementLine(body=[cst.Pass()]))
  assert res is True
  assert p.context.current_stmt_errors == []
  assert p.context.current_stmt_warnings == []


def test_auxiliary_leave_decorator_rename_noncall3() -> None:
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"no_api": "missing"}}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert res is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same(mock_get_hook: MagicMock) -> None:
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same(mock_get_hook: MagicMock) -> None:
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_none(mock_get_hook: MagicMock) -> None:
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.return_value = None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop
