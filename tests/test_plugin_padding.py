"""Test suite for the Plugin Padding module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding


def test_padding_coverage():
  """Verifies the behavior of padding coverage."""
  ctx = MagicMock()
  ctx.semantics = None
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {}}
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": type("Dummy", (), {})}
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics.get_operation.return_value = MagicMock(requires_padding=True)
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("dummy"), args=[])
  transform_padding(node, ctx)
  ctx.lookup_api.return_value = "pad"
  node = cst.Call(func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(cst.Name("b"))])
  transform_padding(node, ctx)
  node = cst.Call(
    func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1"))]))]
  )
  transform_padding(node, ctx)
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(cst.Name("a")),
      cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1")), cst.Element(cst.Integer("1"))])),
    ],
  )
  transform_padding(node, ctx)
