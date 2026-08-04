"""Test suite for the Mlir Generator Naming module."""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, ValueNode


def gen_code(ops: list[OperationNode]) -> str:
  """Helper to generation code."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator()
  return gen.generate(mod).code


def test_naming_from_type_attribute():
  """Verifies the behavior of naming from type attribute."""
  op1 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
    operands=[],
  )
  op2 = OperationNode(
    name="sw.op", operands=[ValueNode(name="%0")], attributes=[AttributeNode(name="type", value='"nop"')]
  )
  code = gen_code([op1, op2])
  assert "_flatten = torch.flatten()" in code
  assert "nop(_flatten)" in code


def test_naming_from_nested_type():
  """Verifies the behavior of naming from nested type."""
  op1 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="type", value='"flax.nnx.Linear"')],
    operands=[],
  )
  op2 = OperationNode(
    name="sw.op", operands=[ValueNode(name="%0")], attributes=[AttributeNode(name="type", value='"nop"')]
  )
  code = gen_code([op1, op2])
  assert "_linear = flax.nnx.Linear()" in code


def test_naming_collision_handling():
  """Verifies the behavior of naming collision handling."""
  op1 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%a")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
    operands=[],
  )
  op2 = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%b")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
    operands=[],
  )
  op3 = OperationNode(
    name="sw.op",
    operands=[ValueNode(name="%a"), ValueNode(name="%b")],
    attributes=[AttributeNode(name="type", value='"nop"')],
  )
  code = gen_code([op1, op2, op3])
  assert "_flatten = torch.flatten()" in code
  assert "_flatten_0 = torch.flatten()" in code
  assert "nop(_flatten, _flatten_0)" in code


def test_naming_fallback():
  """Verifies the behavior of naming fallback."""
  op1 = OperationNode(name="sw.op", results=[ValueNode(name="%a")], attributes=[], operands=[])
  op2 = OperationNode(
    name="sw.op", operands=[ValueNode(name="%a")], attributes=[AttributeNode(name="type", value='"nop"')]
  )
  code = gen_code([op1, op2])
  assert "_a =" in code
  assert "nop(_a)" in code
