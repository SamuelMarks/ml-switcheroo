"""Test suite for the Patcher module."""

import pytest
import typing
import libcst as cst
from ml_switcheroo.core.rewriter.patcher import GraphPatcher, DeleteAction, ReplaceAction
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode


class MockEmitter(PythonSnippetEmitter):
  """Mock Emitter class for testing purposes."""

  def __init__(self) -> None:
    """Init."""
    super().__init__("mock")

  def emit_init(self, node: LogicalNode) -> cst.BaseStatement:
    """Mock implementation of emit initialization."""
    return typing.cast(cst.SimpleStatementLine, cst.parse_statement(f"self.{node.id} = {node.kind}()"))

  def emit_call(self, node: LogicalNode, inputs: list[str], output: str) -> cst.BaseStatement:
    """Mock implementation of emit call."""
    args: str = ", ".join(inputs)
    return typing.cast(cst.SimpleStatementLine, cst.parse_statement(f"{output} = self.{node.id}({args})"))

  def emit_expression(self, node: LogicalNode, inputs: list[str]) -> cst.BaseExpression:
    """Mock implementation of emit expression."""
    args: str = ", ".join(inputs)
    return typing.cast(cst.Call, cst.parse_expression(f"self.{node.id}({args})"))


@pytest.fixture
def emitter() -> MockEmitter:
  """Provides a mock emitter for testing."""
  return MockEmitter()


def test_delete_node(emitter: MockEmitter) -> None:
  """Deletes node."""
  code: str = "\nclass Net:\n    def __init__(self):\n        self.conv = Conv2d()\n        self.bn = BatchNorm()\n"
  module = cst.parse_module(code)
  stmt = typing.cast(
    cst.SimpleStatementLine,
    typing.cast(cst.FunctionDef, typing.cast(cst.ClassDef, module.body[0]).body.body[0]).body.body[1],
  )
  assign = typing.cast(cst.Assign, stmt.body[0])
  provenance: dict[str, typing.Any] = {"bn1": assign}
  plan = [DeleteAction(node_id="bn1")]
  patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
  modified: cst.Module = module.visit(patcher)
  code_out: str = modified.code
  assert "self.conv" in code_out
  assert "self.bn" not in code_out


def test_replace_init_node(emitter: MockEmitter) -> None:
  """Replaces initialization node."""
  code: str = "self.conv = Conv()"
  module = cst.parse_module(code)
  assign_node = typing.cast(cst.Assign, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0])
  provenance: dict[str, typing.Any] = {"c1": assign_node}
  new_node = LogicalNode(id="fused", kind="FusedBlock")
  plan = [ReplaceAction(node_id="c1", new_node=new_node, is_init=True)]
  patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
  modified: cst.Module = module.visit(patcher)
  assert "self.fused = FusedBlock()" in modified.code


def test_replace_call_statement(emitter: MockEmitter) -> None:
  """Replaces call statement."""
  code: str = "x = self.conv(x)"
  module = cst.parse_module(code)
  assign_node = typing.cast(cst.Assign, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0])
  provenance: dict[str, typing.Any] = {"op_conv": assign_node}
  new_node = LogicalNode(id="fused_op", kind="FusedOp")
  plan = [ReplaceAction(node_id="op_conv", new_node=new_node, input_vars=["x", "z"], output_var="y", is_init=False)]
  patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
  modified: cst.Module = module.visit(patcher)
  assert "y = self.fused_op(x, z)" in modified.code


def test_replace_call_expression_nested(emitter: MockEmitter) -> None:
  """Replaces call expression nested."""
  code: str = "return relu(x)"
  module = cst.parse_module(code)
  call_node = typing.cast(cst.Return, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  provenance: dict[str, typing.Any] = {"relu": call_node}
  new_node = LogicalNode(id="fused_relu", kind="FusedOp")
  plan = [ReplaceAction(node_id="relu", new_node=new_node, input_vars=["x"], is_init=False)]
  patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
  modified: cst.Module = module.visit(patcher)
  assert "return self.fused_relu(x)" in modified.code


def test_expression_statement_deletion(emitter: MockEmitter) -> None:
  """Verifies the behavior of expression statement deletion."""
  code: str = "func(x)"
  module = cst.parse_module(code)
  expr_node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  provenance: dict[str, typing.Any] = {"f": expr_node}
  plan = [DeleteAction(node_id="f")]
  patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
  modified: cst.Module = module.visit(patcher)
  assert not modified.body
