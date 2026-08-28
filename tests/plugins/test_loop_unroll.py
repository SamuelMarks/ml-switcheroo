"""Test suite for the Loop Unroll module."""

import pytest
import libcst as cst
from typing import Callable, Dict, Any
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.loop_unroll import transform_loops
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): Code rewriter.
      code (str): The code string.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  new_tree: cst.Module = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory() -> Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: The factory function.
  """
  from ml_switcheroo.core.hooks_registry import clear_hooks, _HOOKS

  clear_hooks()
  _HOOKS["transform_for_loop"] = transform_loops
  _HOOKS["transform_for_loop_static"] = None

  import ml_switcheroo.core.hooks_registry as hr

  hr._PLUGINS_LOADED = True

  mgr: MagicMock = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None

  def get_config(fw: str) -> Dict[str, Any]:
    """Gets configuration.

    Args:
        fw (str): The framework name.

    Returns:
        Dict[str, Any]: The configuration.
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
    cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=False)
    return PivotRewriter(mgr, cfg)

  return create


def test_imperative_passthrough(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of imperative passthrough.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter: PivotRewriter = rewriter_factory("torch")
  code: str = "\nfor i in range(10):\n    print(i)\n"
  result: str = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_functional_range_warning(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of functional range warning.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "\nfor i in range(10):\n    x = x + i\n"
  result: str = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER in result
  assert "JAX requires explicit functional loops" in result


def test_functional_iterator_warning(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of functional iterator warning.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): The factory function.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "\nfor item in my_list:\n    print(item)\n"
  result: str = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert "requires structural rewrite (e.g. `scan`)" in result
