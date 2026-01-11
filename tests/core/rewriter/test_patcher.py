"""
Tests for GraphPatcher CST Logic.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.patcher import (
  GraphPatcher,
  DeleteAction,
  ReplaceAction,
)
from ml_switcheroo.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.compiler.ir import LogicalNode


class MockEmitter(PythonSnippetEmitter):
  """Deterministically mocks code generation."""

  def emit_init(self, node):
    return cst.parse_statement(f"self.{node.id} = {node.kind}()")

  def emit_call(self, node, inputs, output):
    args = ", ".join(inputs)
    return cst.parse_statement(f"{output} = self.{node.id}({args})")

  def emit_expression(self, node, inputs):
    args = ", ".join(inputs)
    return cst.parse_expression(f"self.{node.id}({args})")


@pytest.fixture
def emitter():
  return MockEmitter()


def test_delete_node(emitter):
  """
  Scenario: Delete 'self.bn = BatchNorm()' statement.
  """
  code = """
class Net:
    def __init__(self):
        self.conv = Conv2d()
        self.bn = BatchNorm()
"""
  module = cst.parse_module(code)

  # Extract the Assign node directly from the statement
  stmt = module.body[0].body.body[0].body.body[1]
  assign = stmt.body[0]

  provenance = {"bn1": assign}
  plan = [DeleteAction(node_id="bn1")]

  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)

  # Output: Empty line should be cleaned up
  code_out = modified.code
  assert "self.conv" in code_out
  assert "self.bn" not in code_out


def test_replace_init_node(emitter):
  """
  Scenario: Replace 'self.conv = Conv()' with 'self.fused = Fused()'.
  """
  code = "self.conv = Conv()"
  module = cst.parse_module(code)
  # Extract Assign node
  assign_node = module.body[0].body[0]

  provenance = {"c1": assign_node}
  new_node = LogicalNode(id="fused", kind="FusedBlock")

  plan = [ReplaceAction(node_id="c1", new_node=new_node, is_init=True)]

  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)

  assert "self.fused = FusedBlock()" in modified.code


def test_replace_call_statement(emitter):
  """
  Scenario: Replace 'x = self.conv(x)' (Statement) with 'y = self.fused(x, z)'.
  """
  code = "x = self.conv(x)"
  module = cst.parse_module(code)
  # Extract Assign node
  assign_node = module.body[0].body[0]

  provenance = {"op_conv": assign_node}
  new_node = LogicalNode(id="fused_op", kind="FusedOp")

  plan = [
    ReplaceAction(
      node_id="op_conv",
      new_node=new_node,
      input_vars=["x", "z"],
      output_var="y",
      is_init=False,
    )
  ]

  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)

  assert "y = self.fused_op(x, z)" in modified.code


def test_replace_call_expression_nested(emitter):
  """
  Scenario: Replace 'relu(x)' inside 'return relu(x)'.
  Provenance points to the Call node `relu(x)`.
  """
  code = "return relu(x)"
  module = cst.parse_module(code)
  # Return -> Call -> relu(x)
  call_node = module.body[0].body[0].value

  provenance = {"relu": call_node}
  new_node = LogicalNode(id="fused_relu", kind="FusedOp")

  plan = [ReplaceAction(node_id="relu", new_node=new_node, input_vars=["x"], is_init=False)]

  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)

  # emitted expression: self.fused_relu(x)
  assert "return self.fused_relu(x)" in modified.code


def test_expression_statement_deletion(emitter):
  """
  Scenario: Delete 'func(x)' expression statement.
  Provenance points to Expr node.
  """
  code = "func(x)"
  module = cst.parse_module(code)
  # Extract Expr node from SimpleStatementLine
  expr_node = module.body[0].body[0]

  provenance = {"f": expr_node}
  plan = [DeleteAction(node_id="f")]

  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)

  # SimpleStatementLine removed because it's empty
  assert not modified.body
