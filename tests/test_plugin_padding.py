"""Test suite for the Plugin Padding module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding


def _code(node: cst.CSTNode) -> str:
  """Render a CST node to source code.

  Args:
      node: The CST node to render.

  Returns:
      Rendered Python code string.
  """
  return cst.Module(body=[]).code_for_node(node)


def test_padding_coverage() -> None:
  """Verifies the behavior of padding coverage."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"
  ctx.semantics = None
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {}}
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": type("Dummy", (), {})}
  assert _supports_numpy_padding(ctx) is False
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx.lookup_api.return_value = None
  node: cst.Call = cst.ensure_type(cst.Call(func=cst.Name("dummy"), args=[]), cst.Call)
  assert transform_padding(node, ctx) is node
  ctx.lookup_api.return_value = "pad"
  node2: cst.Call = cst.ensure_type(
    cst.Call(func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(cst.Name("b"))]), cst.Call
  )
  assert transform_padding(node2, ctx) is node2
  node3: cst.BaseExpression = cst.Call(
    func=cst.Name("dummy"), args=[cst.Arg(cst.Name("a")), cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1"))]))]
  )
  assert isinstance(node3, cst.Call)
  assert transform_padding(node3, ctx) is node3
  node4: cst.BaseExpression = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(cst.Name("a")),
      cst.Arg(value=cst.Tuple([cst.Element(cst.Integer("1")), cst.Element(cst.Integer("1"))])),
    ],
  )
  assert isinstance(node4, cst.Call)
  assert transform_padding(node4, ctx) is node4
  node5: cst.BaseExpression = cst.parse_expression("dummy(a, (1, 2, 3, 4))")
  assert isinstance(node5, cst.Call)
  res5 = transform_padding(node5, ctx)
  assert _code(res5) == "pad(a, ((0, 0), (0, 0), (3, 4), (1, 2)))"
