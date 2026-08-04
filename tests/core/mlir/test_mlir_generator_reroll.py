"""Test suite for the Mlir Generator Reroll module."""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.cst import BlockNode, ModuleNode, OperationNode, ValueNode


def gen_code_from_block(ops: list[OperationNode]) -> str:
  """Helper to generation code from block."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_default_rerolling_structure():
  """Verifies the behavior of default rerolling structure."""
  op1 = OperationNode(name="sw.call", results=[ValueNode(name="%0")], operands=[ValueNode(name="%foo")])
  op2 = OperationNode(
    name="sw.call", results=[ValueNode(name="%1")], operands=[ValueNode(name="%bar"), ValueNode(name="%0")]
  )
  code = gen_code_from_block([op1, op2])
  assert "_0 = _foo()" in code
  assert "_bar(_0)" in code


def test_reroll_nested_chain():
  """Verifies the behavior of reroll nested chain."""
  op1 = OperationNode(name="sw.call", results=[ValueNode(name="%a")], operands=[ValueNode(name="%funcA")])
  op2 = OperationNode(
    name="sw.call", results=[ValueNode(name="%b")], operands=[ValueNode(name="%funcB"), ValueNode(name="%a")]
  )
  op3 = OperationNode(
    name="sw.call", results=[ValueNode(name="%c")], operands=[ValueNode(name="%funcC"), ValueNode(name="%b")]
  )
  code = gen_code_from_block([op1, op2, op3])
  assert "_a = _funcA()" in code
  assert "_b = _funcB(_a)" in code
  assert "_funcC(_b)" in code
