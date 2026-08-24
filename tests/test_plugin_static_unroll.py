"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.static_unroll import LoopVarReplacer, unroll_static_loops
from ml_switcheroo.core.hooks import HookContext


def test_loop_var_replacer():
  """Docstring."""
  replacer = LoopVarReplacer("i", 0)

  # match
  node = cst.parse_expression("i")
  new_node = node.visit(replacer)
  assert isinstance(new_node, cst.Integer)
  assert new_node.value == "0"

  # no match
  node = cst.parse_expression("j")
  new_node = node.visit(replacer)
  assert isinstance(new_node, cst.Name)
  assert new_node.value == "j"


def test_unroll_static_loops():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  # Valid static loop
  source = "for i in range(2):\n    x = f(x, i)\n"
  module = cst.parse_module(source)
  for_node = module.body[0]

  result = unroll_static_loops(for_node, ctx)
  assert isinstance(result, cst.FlattenSentinel)
  assert len(result.nodes) == 2

  code0 = cst.Module(body=[result.nodes[0]]).code.strip()
  assert code0 == "x = f(x, 0)"
  code1 = cst.Module(body=[result.nodes[1]]).code.strip()
  assert code1 == "x = f(x, 1)"

  # Loop with no args
  source = "for i in range():\n    pass\n"
  module = cst.parse_module(source)
  for_node = module.body[0]
  assert unroll_static_loops(for_node, ctx) is for_node

  # Loop with non-integer arg
  source = "for i in range(n):\n    pass\n"
  module = cst.parse_module(source)
  for_node = module.body[0]
  assert unroll_static_loops(for_node, ctx) is for_node

  # Loop with large integer arg
  source = "for i in range(100):\n    pass\n"
  module = cst.parse_module(source)
  for_node = module.body[0]
  assert unroll_static_loops(for_node, ctx) is for_node

  # Not a simple name target
  source = "for i, j in range(2):\n    pass\n"
  module = cst.parse_module(source)
  for_node = module.body[0]
  assert unroll_static_loops(for_node, ctx) is for_node

  # Not an indented block (SimpleStatementSuite)
  source = "for i in range(2): pass"
  module = cst.parse_module(source)
  for_node = module.body[0]
  assert unroll_static_loops(for_node, ctx) is for_node
