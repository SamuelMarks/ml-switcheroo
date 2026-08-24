"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.checkpoint_keys import transform_checkpoint_keys


def test_transform_checkpoint_keys_positional_arg():
  """Docstring."""
  # model.load_state_dict(state)
  node = cst.parse_expression("model.load_state_dict(state)")
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  ctx._preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state)"


def test_transform_checkpoint_keys_keyword_arg():
  """Docstring."""
  # model.load_state_dict(state_dict=state_var, strict=True)
  node = cst.parse_expression("model.load_state_dict(state_dict=state_var, strict=True)")
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  ctx._preamble_injector.assert_called()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "KeyMapper.from_torch(state_var)"


def test_transform_checkpoint_keys_no_args():
  """Docstring."""
  # model.load_state_dict()
  node = cst.parse_expression("model.load_state_dict()")
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  # Should return original node
  assert new_node is node
  assert not ctx._preamble_injector.called


def test_transform_checkpoint_keys_already_injected():
  """Docstring."""
  node = cst.parse_expression("model.load_state_dict(state)")
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock())
  ctx.metadata = {"key_mapper_injected": True}

  transform_checkpoint_keys(node, ctx)

  assert not ctx._preamble_injector.called
