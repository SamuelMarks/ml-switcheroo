import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding
from ml_switcheroo.core.hooks import HookContext


def test_padding_coverage():
  ctx = MagicMock()
  # 44: no semantics
  ctx.semantics = None
  assert _supports_numpy_padding(ctx) is False

  # 55: traits is dict without key
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {}}
  assert _supports_numpy_padding(ctx) is False

  # 60: traits is object without attr
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": type("Dummy", (), {})}
  assert _supports_numpy_padding(ctx) is False

  # Let's test transform_padding
  ctx.semantics.get_operation.return_value = MagicMock(requires_padding=True)
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}

  # 86: no target api
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  transform_padding(node, ctx)

  # 97: not tuple
  ctx.lookup_api.return_value = "pad"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(cst.Name("b"))])
  transform_padding(node, ctx)

  # 120: len elements % 2 != 0
  node = cst.Call(
    func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1"))]))]
  )
  transform_padding(node, ctx)

  # 132: pad length <= rank
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(cst.Name("a")),
      cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1")), cst.Element(cst.Integer("1"))])),
    ],
  )
  # to hit 132 it must pass the loop but then something else
  # wait, let's just see if this covers 132.
  transform_padding(node, ctx)
