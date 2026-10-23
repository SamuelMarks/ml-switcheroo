"""Test suite for the GraphPatcher AST surgery module."""

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.differ import GraphDiffer
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.rewriter.patcher import DeleteAction, GraphPatcher, PatchAction, ReplaceAction


class MockEmitter(PythonSnippetEmitter):
  """Mock snippet emitter for deterministic GraphPatcher unit tests."""

  def __init__(self) -> None:
    """Initialize the mock snippet emitter."""
    super().__init__("mock")

  def emit_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
    """Emit a synthetic layer initialization assignment statement.

    Args:
        node: LogicalNode describing the target layer.

    Returns:
        Parsed LibCST SimpleStatementLine for ``self.<id> = <op_type>()``.
    """
    return cst.ensure_type(cst.parse_statement(f"self.{node.id} = {node.op_type}()"), cst.SimpleStatementLine)

  def emit_call(self, node: LogicalNode, inputs: list[str], output: str) -> cst.SimpleStatementLine:
    """Emit a synthetic layer forward call assignment statement.

    Args:
        node: LogicalNode describing the target layer.
        inputs: Ordered list of input variable names.
        output: Output variable name receiving the call result.

    Returns:
        Parsed LibCST SimpleStatementLine for ``<output> = self.<id>(<inputs>)``.
    """
    args = ", ".join(inputs)
    return cst.ensure_type(cst.parse_statement(f"{output} = self.{node.id}({args})"), cst.SimpleStatementLine)

  def emit_expression(self, node: LogicalNode, inputs: list[str]) -> cst.Call:
    """Emit a synthetic layer forward call expression node.

    Args:
        node: LogicalNode describing the target layer.
        inputs: Ordered list of input variable names.

    Returns:
        Parsed LibCST Call expression for ``self.<id>(<inputs>)``.
    """
    args = ", ".join(inputs)
    return cst.ensure_type(cst.parse_expression(f"self.{node.id}({args})"), cst.Call)


@pytest.fixture
def emitter() -> MockEmitter:
  """Provide a MockEmitter fixture for GraphPatcher tests.

  Returns:
      Configured MockEmitter instance.
  """
  return MockEmitter()


def test_delete_node(emitter: MockEmitter) -> None:
  """Verify that DeleteAction removes a targeted layer initialization statement.

  Args:
      emitter: MockEmitter fixture.
  """
  code = """class Net:
    def __init__(self):
        self.conv = Conv2d()
        self.bn = BatchNorm()
"""
  module = cst.parse_module(code)
  cls_def = cst.ensure_type(module.body[0], cst.ClassDef)
  fn_def = cst.ensure_type(cls_def.body.body[0], cst.FunctionDef)
  stmt = cst.ensure_type(fn_def.body.body[1], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  provenance: dict[str, cst.CSTNode] = {"bn1": assign, "unmapped_node": cst.Name("unmapped")}
  plan: list[PatchAction] = [DeleteAction(node_id="bn1"), DeleteAction(node_id="missing_node")]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  code_out = modified.code
  assert "self.conv" in code_out
  assert "self.bn" not in code_out


def test_replace_init_node(emitter: MockEmitter) -> None:
  """Verify that ReplaceAction with is_init=True replaces a layer initialization statement.

  Args:
      emitter: MockEmitter fixture.
  """
  code = "self.conv = Conv()"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  assign_node = cst.ensure_type(stmt.body[0], cst.Assign)
  provenance: dict[str, cst.CSTNode] = {"c1": assign_node}
  new_node = LogicalNode(id="fused", op_type="FusedBlock")
  plan: list[PatchAction] = [ReplaceAction(node_id="c1", new_node=new_node, is_init=True)]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "self.fused = FusedBlock()" in modified.code


def test_replace_call_statement(emitter: MockEmitter) -> None:
  """Verify that ReplaceAction with is_init=False replaces an assignment call statement.

  Args:
      emitter: MockEmitter fixture.
  """
  code = "x = self.conv(x)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  assign_node = cst.ensure_type(stmt.body[0], cst.Assign)
  provenance: dict[str, cst.CSTNode] = {"op_conv": assign_node}
  new_node = LogicalNode(id="fused_op", op_type="FusedOp")
  plan: list[PatchAction] = [
    ReplaceAction(node_id="op_conv", new_node=new_node, input_vars=["x", "z"], output_var="y", is_init=False)
  ]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "y = self.fused_op(x, z)" in modified.code


def test_replace_call_expression_nested(emitter: MockEmitter) -> None:
  """Verify that ReplaceAction replaces a nested Call expression inside a Return statement.

  Args:
      emitter: MockEmitter fixture.
  """
  code = "return relu(x)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  call_node = cst.ensure_type(stmt.body[0], cst.Return).value
  assert call_node is not None
  provenance: dict[str, cst.CSTNode] = {"relu": call_node}
  new_node = LogicalNode(id="fused_relu", op_type="FusedOp")
  plan: list[PatchAction] = [ReplaceAction(node_id="relu", new_node=new_node, input_vars=["x"], is_init=False)]
  patcher = GraphPatcher(plan, provenance, emitter)
  modified = module.visit(patcher)
  assert "return self.fused_relu(x)" in modified.code


def test_expression_statement_deletion_and_replacement(emitter: MockEmitter) -> None:
  """Verify deletion and replacement of standalone Expr statements.

  Args:
      emitter: MockEmitter fixture.
  """
  code = "func(x)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr_node = cst.ensure_type(stmt.body[0], cst.Expr)
  provenance: dict[str, cst.CSTNode] = {"f": expr_node}
  plan_del: list[PatchAction] = [DeleteAction(node_id="f")]
  patcher_del = GraphPatcher(plan_del, provenance, emitter)
  modified_del = module.visit(patcher_del)
  assert not modified_del.body

  new_node = LogicalNode(id="fused_f", op_type="FusedOp")
  plan_rep: list[PatchAction] = [
    ReplaceAction(node_id="f", new_node=new_node, input_vars=["x"], output_var="y", is_init=False)
  ]
  patcher_rep = GraphPatcher(plan_rep, provenance, emitter)
  modified_rep = module.visit(patcher_rep)
  assert "y = self.fused_f(x)" in modified_rep.code


def test_stateful_layer_fusion_init_and_forward_e2e(emitter: MockEmitter) -> None:
  """Verify that GraphExtractor, GraphDiffer, and GraphPatcher patch both __init__ and forward simultaneously.

  Args:
      emitter: MockEmitter fixture.
  """
  code = """class ConvBnNet:
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
"""
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)

  assert "conv" in extractor.node_map
  assert "conv:call" in extractor.node_map
  assert "bn" in extractor.node_map
  assert "bn:call" in extractor.node_map

  target_graph = LogicalGraph(
    name="ConvBnNet",
    nodes={
      "Input_x": LogicalNode(id="Input_x", op_type="Input", attributes={"name": "x"}),
      "fused_conv": LogicalNode(id="fused_conv", op_type="FusedConvBn2d", attributes={"anchor": "conv"}),
      "output": LogicalNode(id="output", op_type="Output"),
    },
    edges=[
      LogicalEdge(source="x", target="fused_conv"),
      LogicalEdge(source="fused_conv", target="output"),
    ],
  )
  differ = GraphDiffer()
  plan = differ.diff(extractor.graph, target_graph)

  patcher = GraphPatcher(plan, extractor.node_map, emitter)
  patched = module.visit(patcher)
  patched_code = patched.code

  assert "self.fused_conv = FusedConvBn2d()" in patched_code
  assert "fused_conv = self.fused_conv(x)" in patched_code
  assert "self.bn" not in patched_code


def test_patcher_fallback_and_edge_branches(emitter: MockEmitter) -> None:
  """Verify fallback branches for unknown PatchAction subclasses and empty replacement statements.

  Args:
      emitter: MockEmitter fixture.
  """

  class CustomAction(PatchAction):
    """Custom no-op PatchAction subclass for testing fallback dispatch."""

  class EmptyEmitter(MockEmitter):
    """Emitter returning empty SimpleStatementLines to exercise fallback return paths."""

    def emit_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
      """Return an empty SimpleStatementLine.

      Args:
          node: LogicalNode to emit.

      Returns:
          Empty SimpleStatementLine.
      """
      return cst.SimpleStatementLine(body=[])

  dummy_assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("a"))], value=cst.Name("b"))
  dummy_expr = cst.Expr(value=cst.Name("a"))
  dummy_call = cst.Call(func=cst.Name("f"))

  provenance: dict[str, cst.CSTNode] = {
    "a1": dummy_assign,
    "e1": dummy_expr,
    "c1": dummy_call,
  }
  plan: list[PatchAction] = [
    CustomAction(node_id="a1"),
    CustomAction(node_id="e1"),
    DeleteAction(node_id="c1"),
  ]
  patcher = GraphPatcher(plan, provenance, emitter)

  assert patcher.leave_Assign(dummy_assign, dummy_assign) is dummy_assign
  assert patcher.leave_Expr(dummy_expr, dummy_expr) is dummy_expr
  assert patcher.leave_Call(dummy_call, dummy_call) is dummy_call

  empty_plan: list[PatchAction] = [
    ReplaceAction(node_id="a1", new_node=LogicalNode(id="x", op_type="Op"), is_init=True),
    ReplaceAction(node_id="e1", new_node=LogicalNode(id="y", op_type="Op"), is_init=True),
  ]
  empty_patcher = GraphPatcher(empty_plan, provenance, EmptyEmitter())
  assert empty_patcher.leave_Assign(dummy_assign, dummy_assign) is dummy_assign
  assert empty_patcher.leave_Expr(dummy_expr, dummy_expr) is dummy_expr

  empty_stmt = cst.SimpleStatementLine(body=[])
  assert patcher._unwrap_stmt_if_nested(dummy_expr, empty_stmt) is empty_stmt
  assert patcher._unwrap_stmt_if_nested(cst.Name("x"), empty_stmt) is empty_stmt
