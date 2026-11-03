"""Unit tests for Phase 12: Synthesizing Target-Framework Forward + Backward VJP Code."""

from __future__ import annotations
from tests.conftest import _test_setattr

import types
from unittest import mock

import libcst as cst
from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.ir_bridge import (
  render_logical_graph_mermaid,
  synthesize_target_backward_code,
)


def _sample_backward_graph() -> LogicalGraph:
  """Construct a sample backward VJP LogicalGraph.

  Returns:
      Populated `LogicalGraph`.
  """
  g = LogicalGraph(name="BackwardNet")
  g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["y"] = LogicalNode(id="y", op_type="Add", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["grad_y"] = LogicalNode(
    id="grad_y", op_type="Constant", attributes={"value": 1.0}, shape_metadata=(2, 4), dtype=DType.float32
  )
  g.nodes["grad_x"] = LogicalNode(id="grad_x", op_type="Add", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["unshaped"] = LogicalNode(id="unshaped", op_type="Identity")
  g.edges.extend(
    [
      LogicalEdge(source="x", target="y"),
      LogicalEdge(source="x", target="y"),
      LogicalEdge(source="grad_y", target="grad_x"),
      LogicalEdge(source="unshaped", target="grad_x"),
    ]
  )
  g.inputs = ["x"]
  g.outputs = ["grad_x"]
  return g


def test_render_logical_graph_mermaid() -> None:
  """Verify render_logical_graph_mermaid emits TensorSpec labels and gradNode styling."""
  g = _sample_backward_graph()
  mermaid = render_logical_graph_mermaid(g, title="Backward VJP")
  assert "graph TD" in mermaid
  assert "TensorSpec((2, 4), float32)" in mermaid
  assert ":::gradNode" in mermaid
  assert "classDef gradNode" in mermaid


def test_synthesize_target_backward_code_live_and_fallback() -> None:
  """Verify synthesize_target_backward_code across target frameworks and PythonBackend fallback."""
  g = _sample_backward_graph()
  stub_ir_to_ast = types.ModuleType("ml_switcheroo_compiler.backends.ir_to_ast")
  _test_setattr(
    stub_ir_to_ast,
    "emit_ir_to_ast",
    lambda graph, fw: cst.parse_module("grad_x = x\nreturn grad_x\n"),
  )
  with mock.patch("importlib.import_module", return_value=stub_ir_to_ast):
    for target in ("jax", "torch", "mlx", "keras", "tensorflow"):
      bwd_code = synthesize_target_backward_code(g, target_fw=target)
      assert "def synthesized_backward_vjp(x):" in bwd_code

  # Empty CST module emitted by emit_ir_to_ast falls back to PythonBackend
  stub_empty_ir_to_ast = types.ModuleType("ml_switcheroo_compiler.backends.ir_to_ast")
  _test_setattr(stub_empty_ir_to_ast, "emit_ir_to_ast", lambda graph, fw: cst.Module(body=[]))
  with mock.patch("importlib.import_module", return_value=stub_empty_ir_to_ast):
    empty_fallback = synthesize_target_backward_code(g, target_fw="jax")
    assert "Synthesized Backward/VJP Graph (jax)" in empty_fallback

  # Test fallback to PythonBackend when emit_ir_to_ast raises
  with mock.patch("importlib.import_module", side_effect=ImportError("no ir_to_ast")):
    fallback_code = synthesize_target_backward_code(g, target_fw="jax")
    assert "Synthesized Backward/VJP Graph (jax)" in fallback_code

  with (
    mock.patch("ml_switcheroo.core.ir_bridge.PythonBackend.compile", side_effect=RuntimeError("backend failure")),
    mock.patch("importlib.import_module", side_effect=ImportError("no ir_to_ast")),
  ):
    err_fallback = synthesize_target_backward_code(g, target_fw="jax")
    assert "def synthesized_backward_vjp(x):" in err_fallback


def test_emit_backward_code_appended_to_target_output() -> None:
  """Verify ASTEngine appends synthesized_backward_code when config.emit_backward_code is True."""
  code = """import torch

def f(x):
    y = torch.relu(x)
    return y
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    enable_on_device_execution=True,
    emit_backward_code=True,
  )
  engine = ASTEngine(config=cfg)
  res = engine.run(code)
  assert res.success is True
  assert res.runtime_trace is not None
  assert res.runtime_trace.synthesized_backward_code != ""
  assert "synthesized_backward_vjp" in res.code or "Synthesized Backward/VJP Graph" in res.code
