"""Test suite for the Static Unroll module."""

from typing import Generator, Union
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.static_unroll import unroll_static_loops
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  try:
    new_tree: cst.Module = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter() -> Generator[PivotRewriter, None, None]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["transform_for_loop"] = unroll_static_loops
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  mgr.get_definition.return_value = None
  mgr.get_framework_config.return_value = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  yield PivotRewriter(mgr, cfg)


def test_unroll_simple_range(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of unroll simple range.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "for i in range(2):\n    print(i)"
  res: str = rewrite_code(rewriter, code)
  assert "for" not in res
  assert "print(0)" in res
  assert "print(1)" in res


def test_unroll_dependency_replacement(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of unroll dependency replacement.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "\nx = 0\nfor i in range(2):\n    x = x + i\n"
  res: str = rewrite_code(rewriter, code)
  assert "x = x + 0" in res
  assert "x = x + 1" in res


def test_ignore_dynamic_range(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore dynamic range.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "for i in range(N):\n    pass"
  res: str = rewrite_code(rewriter, code)
  assert "for i in range(N):" in res


def test_safety_limit(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of safety limit.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "for i in range(100):\n    pass"
  res: str = rewrite_code(rewriter, code)
  assert "range(100)" in res


def test_unroll_value_error(rewriter: PivotRewriter) -> None:
  """Verifies handling of ValueError during integer parsing.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.For = cst.For(
    target=cst.Name("i"),
    iter=cst.Call(func=cst.Name("range"), args=[cst.Arg(cst.Integer("0"))]),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine([cst.Pass()])]),
  )
  from unittest.mock import PropertyMock, patch

  with patch.object(cst.Integer, "value", new_callable=PropertyMock) as mock_val:
    mock_val.return_value = "not an integer"
    res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, rewriter.ctx)
  assert res is node


def test_unroll_target_not_name(rewriter: PivotRewriter) -> None:
  """Verifies behavior when loop target is not a Name.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "for i, j in range(2):\n    pass"
  res: str = rewrite_code(rewriter, code)
  assert "for i, j in range(2):" in res


def test_unroll_body_not_indented(rewriter: PivotRewriter) -> None:
  """Verifies behavior when loop body is not an IndentedBlock.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.For = cst.For(
    target=cst.Name("i"),
    iter=cst.Call(func=cst.Name("range"), args=[cst.Arg(cst.Integer("2"))]),
    body=cst.SimpleStatementSuite(body=[cst.Pass()]),
  )
  res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, rewriter.ctx)
  assert res is node


# --- Merged from test_static_unroll_extra.py ---


def test_static_unroll_iter_not_call() -> None:
  """Verifies the behavior of static unroll iter not call."""
  node: cst.For = cst.For(
    target=cst.Name("i"), iter=cst.List([]), body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])])
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, ctx)
  assert res is node


def test_static_unroll_iter_call_not_range() -> None:
  """Verifies the behavior of static unroll iter call not range."""
  node: cst.For = cst.For(
    target=cst.Name("i"),
    iter=cst.Call(func=cst.Name("enumerate"), args=[]),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.For] = unroll_static_loops(node, ctx)
  assert res is node
