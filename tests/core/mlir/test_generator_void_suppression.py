"""
Tests for Void Assignment Suppression in MLIR Generator.

Verifies that:
1. Unused operations are emitted as Expr statements (no assignment).
2. super().__init__() is emitted as Expr even if it has a result SSA ID.
3. Used operations are still assigned.
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  AttributeNode,
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


def test_suppress_unused_result():
  """
  Scenario: %0 = foo(). %0 is never used.
  Expectation: `foo()` (Expr statement), NOT `_0 = foo()`.
  """
  op = OperationNode(
    name="sw.call",
    results=[ValueNode("%0")],
    operands=[ValueNode("%func")],
  )

  code = gen_code_from_block([op])

  assert "_func()" in code
  assert "=" not in code


def test_assign_used_result():
  """
  Scenario: %0 = foo(). %1 = bar(%0).
  Expectation: `_0 = foo()`, `_bar(_0)`.
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

  # To force assignment of %1, use it in op3
  op3 = OperationNode(name="sw.call", results=[ValueNode("%2")], operands=[ValueNode("%baz"), ValueNode("%1")])

  code = gen_code_from_block([op1, op2, op3])

  # %0 is used by %1
  assert "_0 = _foo()" in code
  # %1 is used by %2
  assert "_1 = _bar(_0)" in code


def test_suppress_super_init():
  """
  Scenario: %res = super().__init__(). even if %res is technically generated.
  Expectation: `super().__init__()` without assignment prefix.
  """
  op_super = OperationNode(name="sw.op", results=[ValueNode("%0")], attributes=[AttributeNode("type", '"super"')])
  op_attr = OperationNode(
    name="sw.getattr",
    results=[ValueNode("%1")],
    operands=[ValueNode("%0")],
    attributes=[AttributeNode("name", '"__init__"')],
  )
  op_call = OperationNode(name="sw.call", results=[ValueNode("%res")], operands=[ValueNode("%1")])

  code = gen_code_from_block([op_super, op_attr, op_call])

  assert "super().__init__()" in code
  assert "=" not in code


def test_super_init_pattern_detection():
  """
  Unit test for `_is_void_call` helper logic using LibCST nodes.
  """
  import libcst as cst

  gen = MlirToPythonGenerator()

  # super().__init__()
  expr = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__")))

  assert gen._is_void_call(expr) is True

  # other.method()
  expr2 = cst.Call(func=cst.Attribute(value=cst.Name("other"), attr=cst.Name("method")))
  assert gen._is_void_call(expr2) is False
