"""
Tests for Explicit Re-rolling (Un-optimization).

Verifies that:
1. Default behavior (inline_expressions=False) forces sequential SSA statements.
2. Nested expressions are broken down.
3. Explicit inlining option (True) restores nested behavior.
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  ValueNode,
)


def gen_code_from_block(ops: list[OperationNode], inline: bool = False) -> str:
  """Helper to generate python code from a list of ops."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator(inline_expressions=inline)
  return gen.generate(mod).code


def test_default_rerolling_structure():
  """
  Scenario: %0 = foo(). %1 = bar(%0).
  %0 is used. %1 is unused.

  With inline_expressions=False (Default):
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

  # Default is False
  code = gen_code_from_block([op1, op2], inline=False)

  assert "_0 = _foo()" in code
  # %1 unused -> expression statement
  assert "_bar(_0)" in code


def test_optional_inlining_behavior():
  """
  Scenario: %0 = foo(). %1 = bar(%0).

  With inline_expressions=True:
    Expect: _1 = bar(foo()) (or bare expression if unused)
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

  code = gen_code_from_block([op1, op2], inline=True)

  # Expect Nesting
  assert "_bar(_foo())" in code
  # _0 should not be assigned explicitly
  assert "_0 =" not in code


def test_reroll_nested_chain():
  """
  Scenario: Chain A -> B -> C.
  Expect assignments for A and B. C is unused.
  """
  op1 = OperationNode(name="sw.call", results=[ValueNode("%a")], operands=[ValueNode("%funcA")])
  op2 = OperationNode(name="sw.call", results=[ValueNode("%b")], operands=[ValueNode("%funcB"), ValueNode("%a")])
  op3 = OperationNode(name="sw.call", results=[ValueNode("%c")], operands=[ValueNode("%funcC"), ValueNode("%b")])

  code = gen_code_from_block([op1, op2, op3], inline=False)

  assert "_a = _funcA()" in code
  assert "_b = _funcB(_a)" in code
  # Unused value -> expression statement
  assert "_funcC(_b)" in code
