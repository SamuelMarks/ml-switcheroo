"""Test module."""

from typing import Dict, Union
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.rewriter.patcher import DeleteAction, GraphPatcher, ReplaceAction


def test_graph_patcher_delete() -> None:
  """Docstring."""
  action: DeleteAction = DeleteAction(node_id="test_node")
  provenance: Dict[str, cst.CSTNode] = {"test_node": cst.Name("test")}
  emitter: MagicMock = MagicMock(spec=PythonSnippetEmitter)

  patcher: GraphPatcher = GraphPatcher(plan=[action], provenance=provenance, emitter=emitter)

  # Test dispatch delete
  node: cst.CSTNode = provenance["test_node"]
  res: Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel] = patcher._handle_node(node, node)
  assert isinstance(res, cst.RemovalSentinel)


def test_graph_patcher_replace_init() -> None:
  """Docstring."""
  logical_node: MagicMock = MagicMock(spec=LogicalNode)
  action: ReplaceAction = ReplaceAction(node_id="test_node", new_node=logical_node, is_init=True)
  provenance: Dict[str, cst.CSTNode] = {"test_node": cst.Name("test")}
  emitter: MagicMock = MagicMock(spec=PythonSnippetEmitter)

  new_stmt: cst.SimpleStatementLine = cst.SimpleStatementLine(
    body=[cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("y"))]
  )
  emitter.emit_init.return_value = new_stmt

  patcher: GraphPatcher = GraphPatcher(plan=[action], provenance=provenance, emitter=emitter)
  node: cst.CSTNode = provenance["test_node"]

  res: Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel] = patcher._handle_node(node, node)
  assert res == new_stmt

  # Nested unwrap
  assign_node: cst.Assign = cst.Assign(targets=[cst.AssignTarget(cst.Name("a"))], value=cst.Name("b"))
  res_nested: Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel] = patcher._unwrap_stmt_if_nested(
    assign_node, new_stmt
  )
  assert isinstance(res_nested, cst.FlattenSentinel)


def test_graph_patcher_replace_call() -> None:
  """Docstring."""
  logical_node: MagicMock = MagicMock(spec=LogicalNode)
  action: ReplaceAction = ReplaceAction(node_id="test_node", new_node=logical_node, is_init=False, input_vars=["a"])
  provenance: Dict[str, cst.CSTNode] = {"test_node": cst.Call(func=cst.Name("test"))}
  emitter: MagicMock = MagicMock(spec=PythonSnippetEmitter)

  new_expr: cst.Name = cst.Name("new_test")
  emitter.emit_expression.return_value = new_expr

  patcher: GraphPatcher = GraphPatcher(plan=[action], provenance=provenance, emitter=emitter)
  node: cst.CSTNode = provenance["test_node"]

  res: Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel] = patcher._handle_node(node, node)
  assert res == new_expr


def test_graph_patcher_replace_call_stmt_like() -> None:
  """Docstring."""
  logical_node: MagicMock = MagicMock(spec=LogicalNode)
  action: ReplaceAction = ReplaceAction(
    node_id="test_node", new_node=logical_node, is_init=False, input_vars=["a"], output_var="y"
  )
  provenance: Dict[str, cst.CSTNode] = {
    "test_node": cst.Assign(targets=[cst.AssignTarget(cst.Name("y"))], value=cst.Name("old"))
  }
  emitter: MagicMock = MagicMock(spec=PythonSnippetEmitter)

  new_stmt: cst.SimpleStatementLine = cst.SimpleStatementLine(
    body=[cst.Assign(targets=[cst.AssignTarget(cst.Name("y"))], value=cst.Name("new"))]
  )
  emitter.emit_call.return_value = new_stmt

  patcher: GraphPatcher = GraphPatcher(plan=[action], provenance=provenance, emitter=emitter)
  node: cst.CSTNode = provenance["test_node"]

  res: Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel] = patcher._handle_node(node, node)
  assert isinstance(res, cst.FlattenSentinel)


def test_graph_patcher_leave_hooks() -> None:
  """Docstring."""
  patcher: GraphPatcher = GraphPatcher([], {}, MagicMock())

  # leave_Assign
  assign: cst.Assign = cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("x"))
  assert patcher.leave_Assign(assign, assign) == assign

  # leave_Expr
  expr: cst.Expr = cst.Expr(value=cst.Name("x"))
  assert patcher.leave_Expr(expr, expr) == expr

  # leave_Call
  call: cst.Call = cst.Call(func=cst.Name("x"))
  assert patcher.leave_Call(call, call) == call

  # leave_SimpleStatementLine
  stmt: cst.SimpleStatementLine = cst.SimpleStatementLine(body=[expr])
  assert patcher.leave_SimpleStatementLine(stmt, stmt) == stmt

  empty_stmt: cst.SimpleStatementLine = cst.SimpleStatementLine(body=[])
  assert isinstance(patcher.leave_SimpleStatementLine(stmt, empty_stmt), cst.RemovalSentinel)
