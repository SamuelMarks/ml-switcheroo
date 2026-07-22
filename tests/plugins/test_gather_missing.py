"""Test suite for the Gather Missing module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.gather import transform_gather
from ml_switcheroo.core.hooks import HookContext


def test_gather_no_target_api():
  """Verifies the behavior of gather no target API."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("gather"))
  res = transform_gather(node, ctx)
  assert res == node


def test_gather_kwargs():
  """Verifies the behavior of gather keyword arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_fw = "jax"
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Integer("1"), keyword=cst.Name("dim")),
      cst.Arg(value=cst.Name("idx"), keyword=cst.Name("index")),
    ],
  )
  transform_gather(node, ctx)


def test_gather_missing_args():
  """Verifies the behavior of gather missing arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_fw = "jax"
  node = cst.Call(func=cst.Name("gather"), args=[cst.Arg(value=cst.Name("x"))])
  res = transform_gather(node, ctx)
  assert res == node
