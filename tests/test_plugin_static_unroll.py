"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.static_unroll import LoopVarReplacer, unroll_static_loops


def test_loop_var_replacer() -> None:
  """Docstring."""
  replacer: LoopVarReplacer = LoopVarReplacer("i", 0)

  # match
  node: cst.BaseExpression = cst.parse_expression("i")
  new_node: cst.CSTNode = node.visit(replacer)
  assert isinstance(new_node, cst.Integer)
  assert getattr(new_node, "value", None) == "0"

  # no match
  node2: cst.BaseExpression = cst.parse_expression("j")
  new_node2: cst.CSTNode = node2.visit(replacer)
  assert isinstance(new_node2, cst.Name)
  assert getattr(new_node2, "value", None) == "j"


def test_unroll_static_loops() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # Valid static loop
  source: str = "for i in range(2):\n    x = f(x, i)\n"
  module: cst.Module = cst.parse_module(source)
  for_node: cst.BaseStatement = getattr(module, "body")[0]

  result: cst.CSTNode = unroll_static_loops(for_node, ctx)
  assert isinstance(result, cst.FlattenSentinel)
  assert len(result.nodes) == 2

  code0: str = cst.Module(body=[result.nodes[0]]).code.strip()
  assert code0 == "x = f(x, 0)"
  code1: str = cst.Module(body=[result.nodes[1]]).code.strip()
  assert code1 == "x = f(x, 1)"

  # Loop with no args
  source_no_args: str = "for i in range():\n    pass\n"
  module_no_args: cst.Module = cst.parse_module(source_no_args)
  for_node_no_args: cst.BaseStatement = getattr(module_no_args, "body")[0]
  assert unroll_static_loops(for_node_no_args, ctx) is for_node_no_args

  # Loop with non-integer arg
  source_non_int: str = "for i in range(n):\n    pass\n"
  module_non_int: cst.Module = cst.parse_module(source_non_int)
  for_node_non_int: cst.BaseStatement = getattr(module_non_int, "body")[0]
  assert unroll_static_loops(for_node_non_int, ctx) is for_node_non_int

  # Loop with large integer arg
  source_large: str = "for i in range(100):\n    pass\n"
  module_large: cst.Module = cst.parse_module(source_large)
  for_node_large: cst.BaseStatement = getattr(module_large, "body")[0]
  assert unroll_static_loops(for_node_large, ctx) is for_node_large

  # Not a simple name target
  source_multi: str = "for i, j in range(2):\n    pass\n"
  module_multi: cst.Module = cst.parse_module(source_multi)
  for_node_multi: cst.BaseStatement = getattr(module_multi, "body")[0]
  assert unroll_static_loops(for_node_multi, ctx) is for_node_multi

  # Not an indented block (SimpleStatementSuite)
  source_simple: str = "for i in range(2): pass"
  module_simple: cst.Module = cst.parse_module(source_simple)
  for_node_simple: cst.BaseStatement = getattr(module_simple, "body")[0]
  assert unroll_static_loops(for_node_simple, ctx) is for_node_simple
