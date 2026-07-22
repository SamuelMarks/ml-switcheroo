"""Test suite for the Auto Fsdp Wrapper Missing Api module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding
from ml_switcheroo.core.hooks import HookContext


def test_auto_fsdp_wrapper_no_api():
  """Verifies the behavior of auto FSDP wrapper no API."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(effective_target="torch"))
  ctx.current_op_id = "Conv2d"
  ctx.semantics.get_operation.return_value = MagicMock(sharding_supported=True)
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"sharding_wrapper_api": None}}
  result = wrap_with_sharding(node, ctx)
  assert result is node
