"""Test suite for the Mlir Generator Void Suppression module."""

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code_from_block(ops: list[OperationNode]) -> str:
  """Execute `ModuleNode` to support gen code from block in MLIR generator void suppression tests.

  Args:
      ops: Input `ops` parameter value (`list[OperationNode]`).

  Returns:
      Resulting `str` from gen code from block.
  """
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_suppress_unused_result() -> None:
  """Verify `OperationNode` emits `_func()` for suppress unused result."""
  op = OperationNode(name="sw.call", results=[ValueNode(name="%0")], operands=[ValueNode(name="%func")])
  code: str = gen_code_from_block([op])
  assert "_func()" in code
  assert "=" not in code


def test_assign_used_result() -> None:
  """Verify `OperationNode` emits `_0 = _foo()` for assign used result."""
  op1 = OperationNode(name="sw.call", results=[ValueNode(name="%0")], operands=[ValueNode(name="%foo")])
  op2 = OperationNode(
    name="sw.call", results=[ValueNode(name="%1")], operands=[ValueNode(name="%bar"), ValueNode(name="%0")]
  )
  op3 = OperationNode(
    name="sw.call", results=[ValueNode(name="%2")], operands=[ValueNode(name="%baz"), ValueNode(name="%1")]
  )
  code: str = gen_code_from_block([op1, op2, op3])
  assert "_0 = _foo()" in code
  assert "_1 = _bar(_0)" in code


def test_suppress_super_init() -> None:
  """Verify `OperationNode` emits `super().__init__()` for suppress super init."""
  op_super = OperationNode(
    name="sw.op", results=[ValueNode(name="%0")], attributes=[AttributeNode(name="type", value='"super"')]
  )
  op_attr = OperationNode(
    name="sw.getattr",
    results=[ValueNode(name="%1")],
    operands=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="name", value='"__init__"')],
  )
  op_call = OperationNode(name="sw.call", results=[ValueNode(name="%res")], operands=[ValueNode(name="%1")])
  code: str = gen_code_from_block([op_super, op_attr, op_call])
  assert "super().__init__()" in code
  assert "=" not in code


def test_super_init_pattern_detection() -> None:
  """Verify `MlirToPythonGenerator` handles super init pattern detection as expected."""
  import libcst as cst

  gen = MlirToPythonGenerator()
  expr = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__")))
  assert gen._is_void_call(expr) is True
  expr2 = cst.Call(func=cst.Attribute(value=cst.Name("other"), attr=cst.Name("method")))
  assert gen._is_void_call(expr2) is False
