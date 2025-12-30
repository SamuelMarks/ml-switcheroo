"""
Tests for Semantic Variable Naming using Type Hints.

Verifies that:
1. Operations with 'type' attributes generate readable variable names (e.g. _flatten).
2. Numeric collisions are handled safely (e.g. _flatten vs _flatten_0).
3. Fallback logic preserves SSA numbering if no type is available.
"""

from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.nodes import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  ValueNode,
)


def gen_code(ops: list[OperationNode]) -> str:
  """Generates Python code from a list of ops."""
  mod = ModuleNode(body=BlockNode(label="", operations=ops))
  gen = MlirToPythonGenerator(inline_expressions=False)
  return gen.generate(mod).code


def test_naming_from_type_attribute():
  """
  Scenario: %0 = sw.op {type="torch.flatten"}
  Expectation: _flatten = torch.flatten(...)
  """
  # Must use result in another op to force assignment
  op1 = OperationNode(
    name="sw.op", results=[ValueNode("%0")], attributes=[AttributeNode("type", '"torch.flatten"')], operands=[]
  )
  op2 = OperationNode(name="sw.op", operands=[ValueNode("%0")], attributes=[AttributeNode("type", '"nop"')])

  code = gen_code([op1, op2])

  assert "_flatten = torch.flatten()" in code
  # Verify we replaced %0
  assert "nop(_flatten)" in code


def test_naming_from_nested_type():
  """
  Scenario: %0 = sw.op {type="flax.nnx.Linear"}
  Expectation: _linear = flax.nnx.Linear(...)
  """
  op1 = OperationNode(
    name="sw.op", results=[ValueNode("%0")], attributes=[AttributeNode("type", '"flax.nnx.Linear"')], operands=[]
  )
  op2 = OperationNode(name="sw.op", operands=[ValueNode("%0")], attributes=[AttributeNode("type", '"nop"')])

  code = gen_code([op1, op2])

  assert "_linear = flax.nnx.Linear()" in code


def test_naming_collision_handling():
  """
  Scenario: Two different flattens.
  Expectation: _flatten and _flatten_0.
  """
  op1 = OperationNode(
    name="sw.op", results=[ValueNode("%a")], attributes=[AttributeNode("type", '"torch.flatten"')], operands=[]
  )
  op2 = OperationNode(
    name="sw.op", results=[ValueNode("%b")], attributes=[AttributeNode("type", '"torch.flatten"')], operands=[]
  )
  op3 = OperationNode(
    name="sw.op", operands=[ValueNode("%a"), ValueNode("%b")], attributes=[AttributeNode("type", '"nop"')]
  )

  code = gen_code([op1, op2, op3])

  assert "_flatten = torch.flatten()" in code
  assert "_flatten_0 = torch.flatten()" in code
  assert "nop(_flatten, _flatten_0)" in code


def test_naming_fallback():
  """
  Scenario: No type attribute.
  Expectation: SSA ID preservation (e.g. _a).
  """
  op1 = OperationNode(
    name="sw.op",
    results=[ValueNode("%a")],
    attributes=[],  # No type
    operands=[],
  )
  # Use 'sw.op' as consumer to prevent Statement Fusion (which happens on sw.return)
  # Usage count needs to be > 1 to prevent inlining in inline_expressions=False mode?
  # No, default is re-rolled.
  # However, Statement Fusion logic in generator aggressively fuses single-use if consumer is
  # 'setattr' or 'return'.
  # We use 'sw.op' ("nop") as consumer to ensure assignment generation.
  op2 = OperationNode(name="sw.op", operands=[ValueNode("%a")], attributes=[AttributeNode("type", '"nop"')])

  code = gen_code([op1, op2])

  # Default fallback replaces % with _ leading to _a
  assert "_a =" in code
  assert "nop(_a)" in code
