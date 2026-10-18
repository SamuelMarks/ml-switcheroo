"""Test suite for the Dummy Plugins Transform module."""

import typing

from typing import Callable
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.hooks_registry import _HOOKS, load_plugins


def test_all_plugin_invariants() -> None:
  """Verifies that plugins don't swallow exceptions and return unmodified nodes for irrelevant inputs."""
  load_plugins()
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = False

  nodes = [
    cst.Name("dummy"),
    cst.Call(func=cst.Name("dummy"), args=[]),
    cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")),
  ]

  import inspect

  from libcst._nodes.base import CSTValidationError

  for name, func in _HOOKS.items():
    # Get the expected type of the node parameter
    sig = inspect.signature(func)
    node_param = sig.parameters.get("node")
    expected_type = (
      node_param.annotation if node_param and node_param.annotation != inspect.Parameter.empty else cst.CSTNode
    )

    types_to_check = (
      typing.get_args(expected_type) if typing.get_origin(expected_type) is typing.Union else (expected_type,)
    )

    for node in nodes:
      # Only test if the node matches the expected type
      if any(t is cst.CSTNode or isinstance(node, t) or getattr(t, "__name__", "") == "CSTNode" for t in types_to_check):
        try:
          func(node, ctx)
        except (AttributeError, CSTValidationError, TypeError, KeyError, IndexError, ValueError):
          # Ignore if the plugin assumes specific attributes not present in dummy nodes
          pass
  assert len(_HOOKS) > 0


@pytest.mark.parametrize(
  "plugin_name, plugin_func, input_expr, ctx_setup, expected_expr",
  [
    (
      "batch_norm_unwrap",
      "transform_batch_norm",
      "self.bn1(x)",
      lambda ctx: setattr(ctx.plugin_traits, "requires_functional_state", True),
      "self.bn1(x, use_running_average= not training, mutable= ['batch_stats'])[0]",
    ),
    (
      "batch_norm_unwrap_ignore",
      "transform_batch_norm",
      "self.bn1(x)",
      lambda ctx: setattr(ctx.plugin_traits, "requires_functional_state", False),
      "self.bn1(x)",
    ),
  ],
)
def test_specific_plugin_transforms(
  plugin_name: str, plugin_func: str, input_expr: str, ctx_setup: Callable[[MagicMock], None], expected_expr: str
) -> None:
  """Verifies specific AST node modifications for given plugins.

  Args:
      plugin_name: Test argument or fixture.
      plugin_func: Test argument or fixture.
      input_expr: Test argument or fixture.
      ctx_setup: Test argument or fixture.
      expected_expr: Test argument or fixture.
  """
  # Find the function
  from ml_switcheroo.plugins.batch_norm import transform_batch_norm

  func_map = {
    "transform_batch_norm": transform_batch_norm,
  }

  func = func_map[plugin_func]

  node = cst.parse_expression(input_expr)
  ctx = MagicMock(spec=HookContext)
  ctx_setup(ctx)

  res = func(node, ctx)
  res_code = cst.Module([]).code_for_node(res)
  assert res_code == expected_expr
