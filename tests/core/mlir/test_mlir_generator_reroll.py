"""Test suite for the Mlir Generator Reroll module."""

from ml_switcheroo.core.mlir.cst import BlockNode, ModuleNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code_from_block(ops: list[OperationNode]) -> str:
  """Execute `ModuleNode` to support gen code from block in MLIR generator reroll tests.

  Args:
      ops: Input `ops` parameter value (`list[OperationNode]`).

  Returns:
      Resulting `str` from gen code from block.
  """
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_default_rerolling_structure() -> None:
  """Verify `OperationNode` emits `_0 = _foo()` for default rerolling structure."""
  op1 = OperationNode(name="sw.call", results=[ValueNode(name="%0")], operands=[ValueNode(name="%foo")])
  op2 = OperationNode(
    name="sw.call", results=[ValueNode(name="%1")], operands=[ValueNode(name="%bar"), ValueNode(name="%0")]
  )
  code: str = gen_code_from_block([op1, op2])
  assert "_0 = _foo()" in code
  assert "_bar(_0)" in code


def test_reroll_nested_chain() -> None:
  """Verify `OperationNode` emits `_a = _funcA()` for reroll nested chain."""
  op1 = OperationNode(name="sw.call", results=[ValueNode(name="%a")], operands=[ValueNode(name="%funcA")])
  op2 = OperationNode(
    name="sw.call", results=[ValueNode(name="%b")], operands=[ValueNode(name="%funcB"), ValueNode(name="%a")]
  )
  op3 = OperationNode(
    name="sw.call", results=[ValueNode(name="%c")], operands=[ValueNode(name="%funcC"), ValueNode(name="%b")]
  )
  code: str = gen_code_from_block([op1, op2, op3])
  assert "_a = _funcA()" in code
  assert "_b = _funcB(_a)" in code
  assert "_funcC(_b)" in code
