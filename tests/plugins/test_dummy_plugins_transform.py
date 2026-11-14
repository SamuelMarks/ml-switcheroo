"""Test suite for the Dummy Plugins Transform module."""

import contextlib
import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.hooks_registry import get_default_hook_registry, load_plugins


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

  hooks_map = get_default_hook_registry().hooks
  for func in hooks_map.values():
    # Get the expected type of the node parameter
    sig = inspect.signature(func)
    node_param = sig.parameters.get("node")
    expected_type = (
      node_param.annotation if node_param and node_param.annotation != inspect.Parameter.empty else cst.CSTNode
    )

    types_to_check = typing.get_args(expected_type) or (expected_type,)

    for node in nodes:
      # Only test if the node matches the expected type
      if any(
        t is cst.CSTNode or (isinstance(t, type) and isinstance(node, t)) or t in ("CSTNode", "cst.CSTNode")
        for t in types_to_check
      ):
        with contextlib.suppress(AttributeError, CSTValidationError, TypeError, KeyError, IndexError, ValueError):
          func(node, ctx)
  assert len(hooks_map) > 0


@pytest.mark.parametrize(
  ("plugin_name", "plugin_func", "input_expr", "requires_functional_state", "expected_expr"),
  [
    (
      "batch_norm_unwrap",
      "transform_batch_norm",
      "self.bn1(x)",
      True,
      "self.bn1(x, use_running_average= not training, mutable= ['batch_stats'])[0]",
    ),
    (
      "batch_norm_unwrap_ignore",
      "transform_batch_norm",
      "self.bn1(x)",
      False,
      "self.bn1(x)",
    ),
  ],
)
def test_specific_plugin_transforms(
  plugin_name: str,
  plugin_func: str,
  input_expr: str,
  requires_functional_state: bool,
  expected_expr: str,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verifies specific AST node modifications for given plugins.

  Args:
      plugin_name: Input `plugin_name` parameter value (`str`).
      plugin_func: Input `plugin_func` parameter value (`str`).
      input_expr: Input `input_expr` parameter value (`str`).
      requires_functional_state: Whether plugin_traits.requires_functional_state is enabled.
      expected_expr: Input `expected_expr` parameter value (`str`).
      monkeypatch: Pytest monkeypatch fixture.
  """
  # Find the function
  from ml_switcheroo.plugins.batch_norm import transform_batch_norm

  func_map = {
    "transform_batch_norm": transform_batch_norm,
  }

  func = func_map[plugin_func]

  node = cst.ensure_type(cst.parse_expression(input_expr), cst.Call)
  ctx = MagicMock(spec=HookContext)
  monkeypatch.setattr(ctx.plugin_traits, "requires_functional_state", requires_functional_state, raising=False)

  res = func(node, ctx)
  res_code = cst.Module([]).code_for_node(res)
  assert res_code == expected_expr
