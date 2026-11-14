"""Test suite for plugin checkpoint keys."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.checkpoint_keys import transform_checkpoint_keys


def test_transform_checkpoint_keys_positional_arg() -> None:
  """Verify `ensure_type` handles transform checkpoint keys positional arg as expected."""
  # model.load_state_dict(state)
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state)"), cst.Call)
  preamble_injector = MagicMock()
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=preamble_injector)

  new_node = cst.ensure_type(transform_checkpoint_keys(node, ctx), cst.Call)

  preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state)"


def test_transform_checkpoint_keys_keyword_arg() -> None:
  """Verify `ensure_type` handles transform checkpoint keys keyword arg as expected."""
  # model.load_state_dict(state_dict=state_var, strict=True)
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state_dict=state_var, strict=True)"), cst.Call)
  preamble_injector = MagicMock()
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=preamble_injector)

  new_node = cst.ensure_type(transform_checkpoint_keys(node, ctx), cst.Call)

  preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state_var)"


def test_transform_checkpoint_keys_no_args() -> None:
  """Verify `ensure_type` handles transform checkpoint keys no args as expected."""
  # model.load_state_dict()
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict()"), cst.Call)
  preamble_injector = MagicMock()
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=preamble_injector)

  new_node = transform_checkpoint_keys(node, ctx)

  # Should return original node
  assert new_node is node
  assert preamble_injector.called is False


def test_transform_checkpoint_keys_already_injected() -> None:
  """Verify `ensure_type` handles transform checkpoint keys already injected as expected."""
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state)"), cst.Call)
  preamble_injector = MagicMock()
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=preamble_injector)
  ctx.metadata = {"key_mapper_injected": True}

  transform_checkpoint_keys(node, ctx)

  assert preamble_injector.called is False
