"""
Tests for Explicit Re-rolling (Default Behavior).

Verifies that:
1. Sequential SSA statements are generated for nested operations.
2. Nested expressions are broken down.
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  ValueNode,
)


def gen_code_from_block(ops: list[OperationNode]) -> str:
  """Helper to generate python code from a list of ops."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_default_rerolling_structure():
  """
  Scenario: %0 = foo(). %1 = bar(%0).
  %0 is used. %1 is unused.

  Expect: _0 = foo(); bar(_0)
  """
  op1 = OperationNode(
    name="sw.call",
    results=[ValueNode("%0")],
    operands=[ValueNode("%foo")],
  )
  op2 = OperationNode(
    name="sw.call",
    results=[ValueNode("%1")],
    operands=[ValueNode("%bar"), ValueNode("%0")],
  )

  code = gen_code_from_block([op1, op2])

  assert "_0 = _foo()" in code
  # %1 unused -> expression statement
  assert "_bar(_0)" in code


def test_reroll_nested_chain():
  """
  Scenario: Chain A -> B -> C.
  Expect assignments for A and B. C is unused.
  """
  op1 = OperationNode(name="sw.call", results=[ValueNode("%a")], operands=[ValueNode("%funcA")])
  op2 = OperationNode(
    name="sw.call",
    results=[ValueNode("%b")],
    operands=[ValueNode("%funcB"), ValueNode("%a")],
  )
  op3 = OperationNode(
    name="sw.call",
    results=[ValueNode("%c")],
    operands=[ValueNode("%funcC"), ValueNode("%b")],
  )

  code = gen_code_from_block([op1, op2, op3])

  assert "_a = _funcA()" in code
  assert "_b = _funcB(_a)" in code
  # Unused value -> expression statement
  assert "_funcC(_b)" in code
