"""Test suite for plugin checkpoint keys."""

import typing

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.checkpoint_keys import transform_checkpoint_keys


def test_transform_checkpoint_keys_positional_arg() -> None:
  """Verify transform checkpoint keys positional arg behavior."""
  # model.load_state_dict(state)
  node: typing.Any = cst.parse_expression("model.load_state_dict(state)")
  ctx: typing.Any = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node: typing.Any = transform_checkpoint_keys(node, ctx)

  ctx._preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code: typing.Any = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state)"


def test_transform_checkpoint_keys_keyword_arg() -> None:
  """Verify transform checkpoint keys keyword arg behavior."""
  # model.load_state_dict(state_dict=state_var, strict=True)
  node: typing.Any = cst.parse_expression("model.load_state_dict(state_dict=state_var, strict=True)")
  ctx: typing.Any = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node: typing.Any = transform_checkpoint_keys(node, ctx)

  ctx._preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code: typing.Any = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state_var)"


def test_transform_checkpoint_keys_no_args() -> None:
  """Verify transform checkpoint keys no args behavior."""
  # model.load_state_dict()
  node: typing.Any = cst.parse_expression("model.load_state_dict()")
  ctx: typing.Any = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node: typing.Any = transform_checkpoint_keys(node, ctx)

  # Should return original node
  assert new_node is node
  assert not getattr(ctx._preamble_injector, "called")


def test_transform_checkpoint_keys_already_injected() -> None:
  """Verify transform checkpoint keys already injected behavior."""
  node: typing.Any = cst.parse_expression("model.load_state_dict(state)")
  ctx: typing.Any = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock())
  ctx.metadata = {"key_mapper_injected": True}

  transform_checkpoint_keys(node, ctx)

  assert not getattr(ctx._preamble_injector, "called")
