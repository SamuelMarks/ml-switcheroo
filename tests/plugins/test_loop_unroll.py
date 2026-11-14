"""Test suite for the Loop Unroll module."""

import typing
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
  from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook

  clear_hooks(mark_loaded=True)
  register_hook("transform_for_loop")(transform_loops)

  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None

  def get_config(fw: str) -> dict[str, PluginTraits]:
    """Gets configuration.

    Args:
        fw (str): The framework name.

    Returns:
        dict[str, PluginTraits]: The configuration.
    """
    if fw == "torch":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=False)}
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=True)}
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
  """Verify `rewriter_factory` emits `for i in range(10):` for imperative passthrough.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter = rewriter_factory("torch")
  code = "\nfor i in range(10):\n    print(i)\n"
  result = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_functional_range_warning(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `for i in range(10):` for functional range warning.

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
  """Verify `rewriter_factory` handles functional iterator warning as expected.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter = rewriter_factory("jax")
  code = "\nfor item in my_list:\n    print(item)\n"
  result = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert "requires structural rewrite (e.g. `scan`)" in result
