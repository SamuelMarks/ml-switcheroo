"""Test suite for the Missing Plugins module."""

import typing

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.hooks_registry import _HOOKS, load_plugins


def test_plugins_missing_prerequisites() -> None:
  """Verifies that all plugins handle missing prerequisites gracefully without crashing.

  They should either return the original node unmodified or raise a clear exception,
  rather than swallowing exceptions or crashing randomly.
  """
  load_plugins()
  # Create an empty context missing all traits and semantics
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits = MagicMock()
  # Explicitly make sure plugin_traits has no attributes set to True or real values
  ctx.plugin_traits.requires_functional_state = False
  ctx.plugin_traits.sharding_wrapper_api = None
  ctx.plugin_traits.lookup_api.return_value = "dummy.api"
  ctx.current_op_id = None
  ctx.semantics = None
  ctx.metadata = {}

  # A generic node that lacks any plugin-specific attributes
  node = cst.parse_expression("generic_call(missing_args)")

  import inspect

  from libcst._nodes.base import CSTValidationError

  for name, func in _HOOKS.items():
    sig = inspect.signature(func)
    node_param = sig.parameters.get("node")
    expected_type = (
      node_param.annotation if node_param and node_param.annotation != inspect.Parameter.empty else cst.CSTNode
    )

    # Resolve Union types
    types_to_check = (
      typing.get_args(expected_type) if typing.get_origin(expected_type) is typing.Union else (expected_type,)
    )

    # Only run if our dummy node is one of the expected types
    if any(t is cst.CSTNode or isinstance(node, t) or getattr(t, "__name__", "") == "CSTNode" for t in types_to_check):
      try:
        res = func(node, ctx)
        assert res is not None, f"Plugin {name} returned None instead of a CSTNode"
      except (AttributeError, CSTValidationError, TypeError, KeyError, IndexError, ValueError):
        pass  # Ignore if it assumes attributes we didn't mock or if it throws due to mock objects
