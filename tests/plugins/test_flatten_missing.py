"""Test suite for the Flatten Missing module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.core.hooks import HookContext


def test_flatten_unhandled_fw():
  """Verifies the behavior of flatten unhandled framework."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "unknown"
  ctx.current_op_id = "Flatten"
  node = cst.Call(func=cst.Name("flatten"))
  res = transform_flatten(node, ctx)
  assert res == node
