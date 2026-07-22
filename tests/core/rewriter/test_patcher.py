"""Test suite for the Patcher module."""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.patcher import GraphPatcher, DeleteAction, ReplaceAction
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode


class MockEmitter(PythonSnippetEmitter):
  """Mock Emitter class for testing purposes."""

  def emit_init(self, node):
    """Mock implementation of emit initialization."""
    return cst.parse_statement(f"self.{node.id} = {node.kind}()")

  def emit_call(self, node, inputs, output):
    """Mock implementation of emit call."""
    args = ", ".join(inputs)
    return cst.parse_statement(f"{output} = self.{node.id}({args})")

  def emit_expression(self, node, inputs):
    """Mock implementation of emit expression."""
    args = ", ".join(inputs)
    return cst.parse_expression(f"self.{node.id}({args})")


@pytest.fixture
def emitter():
  """Provides a mock emitter for testing."""
  return MockEmitter()


def test_delete_node(emitter):
  """Deletes node."""
  code = "\nclass Net:\n    def __init__(self):\n        self.conv = Conv2d()\n        self.bn = BatchNorm()\n"
  module = cst.parse_module(code)
  stmt = module.body[0].body.body[0].body.body[1]
  assign = stmt.body[0]
  provenance = {"bn1": assign}
  plan = [DeleteAction(node_id="bn1")]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  code_out = modified.code
  assert "self.conv" in code_out
  assert "self.bn" not in code_out


def test_replace_init_node(emitter):
  """Replaces initialization node."""
  code = "self.conv = Conv()"
  module = cst.parse_module(code)
  assign_node = module.body[0].body[0]
  provenance = {"c1": assign_node}
  new_node = LogicalNode(id="fused", kind="FusedBlock")
  plan = [ReplaceAction(node_id="c1", new_node=new_node, is_init=True)]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "self.fused = FusedBlock()" in modified.code


def test_replace_call_statement(emitter):
  """Replaces call statement."""
  code = "x = self.conv(x)"
  module = cst.parse_module(code)
  assign_node = module.body[0].body[0]
  provenance = {"op_conv": assign_node}
  new_node = LogicalNode(id="fused_op", kind="FusedOp")
  plan = [ReplaceAction(node_id="op_conv", new_node=new_node, input_vars=["x", "z"], output_var="y", is_init=False)]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "y = self.fused_op(x, z)" in modified.code


def test_replace_call_expression_nested(emitter):
  """Replaces call expression nested."""
  code = "return relu(x)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value
  provenance = {"relu": call_node}
  new_node = LogicalNode(id="fused_relu", kind="FusedOp")
  plan = [ReplaceAction(node_id="relu", new_node=new_node, input_vars=["x"], is_init=False)]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "return self.fused_relu(x)" in modified.code


def test_expression_statement_deletion(emitter):
  """Verifies the behavior of expression statement deletion."""
  code = "func(x)"
  module = cst.parse_module(code)
  expr_node = module.body[0].body[0]
  provenance = {"f": expr_node}
  plan = [DeleteAction(node_id="f")]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert not modified.body
