"""Test suite for the Jax Decompose module."""

from typing import Union
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax


def test_jax_decompose_plugin() -> None:
  """Verifies the behavior of JAX decompose plugin."""
  node: cst.Call = cst.Call(func=cst.Name("UnsupportedOp"), args=[])
  mock_config: RuntimeConfig = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics: MagicMock = MagicMock()
  ctx: HookContext = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Hardswish"
  result: Union[cst.CSTNode, cst.Call] = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "hardswish"
  assert isinstance(result.func.value, cst.Attribute)
  assert result.func.value.attr.value == "numpy"


def test_jax_decompose_plugin_no_op_id() -> None:
  """Verifies the behavior of JAX decompose plugin no op id."""
  node: cst.Call = cst.Call(func=cst.Name("UnsupportedOp"), args=[])
  mock_config: RuntimeConfig = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics: MagicMock = MagicMock()
  ctx: HookContext = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None
  result: Union[cst.CSTNode, cst.Call] = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "unknownop"
