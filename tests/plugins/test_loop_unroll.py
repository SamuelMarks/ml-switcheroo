"""Test suite for the Loop Unroll module."""

import typing
from pydantic import JsonValue

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.plugins.loop_unroll import transform_loops
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): Code rewriter.
      code (str): The code string.

  Returns:
      str: The rewritten code string.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: The factory function.
  """
  from ml_switcheroo.core.hooks_registry import _HOOKS, clear_hooks

  clear_hooks()
  getattr(getattr({"v": _HOOKS}, "__getitem__")("v"), "__setitem__")(
    "transform_for_loop", getattr({"v": transform_loops}, "__getitem__")("v")
  )

  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = True

  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None

  def get_config(fw: str) -> dict[str, JsonValue]:
    """Gets configuration.

    Args:
        fw (str): The framework name.

    Returns:
        dict[str, JsonValue]: The configuration.
    """
    if fw == "torch":
      _ret_v_1: dict[str, JsonValue] = getattr(
        {
          "v": {
            "plugin_traits": getattr(
              {
                "v": getattr({"v": PluginTraits}, "__getitem__")("v")(
                  requires_functional_control_flow=getattr({"v": False}, "__getitem__")("v")
                )
              },
              "__getitem__",
            )("v")
          }
        },
        "__getitem__",
      )("v")
      return _ret_v_1
    if fw == "jax":
      _ret_v_2: dict[str, JsonValue] = getattr(
        {
          "v": {
            "plugin_traits": getattr(
              {
                "v": getattr({"v": PluginTraits}, "__getitem__")("v")(
                  requires_functional_control_flow=getattr({"v": True}, "__getitem__")("v")
                )
              },
              "__getitem__",
            )("v")
          }
        },
        "__getitem__",
      )("v")
      return _ret_v_2
    return {}

  mgr.get_framework_config.side_effect = get_config

  def create(target: str) -> PivotRewriter:
    """Creates rewriter.

    Args:
        target (str): Target framework string.

    Returns:
        PivotRewriter: The rewriter.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=False)
    return PivotRewriter(mgr, cfg)

  return create


def test_imperative_passthrough(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of imperative passthrough.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter = rewriter_factory("torch")
  code = "\nfor i in range(10):\n    print(i)\n"
  result = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_functional_range_warning(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of functional range warning.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter = rewriter_factory("jax")
  code = "\nfor i in range(10):\n    x = x + i\n"
  result = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER in result
  assert "JAX requires explicit functional loops" in result


def test_functional_iterator_warning(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of functional iterator warning.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter = rewriter_factory("jax")
  code = "\nfor item in my_list:\n    print(item)\n"
  result = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert "requires structural rewrite (e.g. `scan`)" in result
