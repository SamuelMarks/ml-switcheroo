"""Unit tests for Phase 10: Zero-Copy `ml-switcheroo-ir` (`LogicalGraph`) Bridge."""

from __future__ import annotations

import types
from unittest import mock

import libcst as cst
from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.config import (
  ConversionMode,
  ExecutionBackend,
  LossReduction,
  OptimizationLevel,
  RuntimeConfig,
)
from ml_switcheroo.core.ir_bridge import (
  _resolve_ir_dtype,
  build_hydrated_forward_graph,
  synthesize_backward_vjp_graph,
)
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry
from ml_switcheroo.core.runtime_tracer import _invoke_codegen, run_on_device_trace


def test_build_hydrated_forward_graph_and_autodiff_grad() -> None:
  """Verify build_hydrated_forward_graph and synthesize_backward_vjp_graph on shared LogicalGraph."""
  code = """def forward(x):
    y = x + x
    return y
"""
  tree = cst.parse_module(code)
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    mode=ConversionMode.OPTIMIZED,
    opt_level=OptimizationLevel.O2,
    execution_backend=ExecutionBackend.WEBGPU,
    loss_reduction=LossReduction.MEAN,
  )
  tel = {
    "y": NodeRuntimeTelemetry(
      node_id="y",
      var_name="y",
      op_type="Add",
      lineno=2,
      output_shape=[2, 4],
      dtype="float32",
    )
  }
  fwd_g, passes = build_hydrated_forward_graph(
    code=code,
    tree=tree,
    input_shapes=[(2, 4)],
    output_shape=(2, 4),
    dtype="float32",
    node_telemetry=tel,
    observed_shapes={"y": [2, 4]},
    config=cfg,
  )
  assert isinstance(fwd_g, LogicalGraph)
  assert len(fwd_g.nodes) >= 1
  assert len(passes) >= 2

  bwd_g = synthesize_backward_vjp_graph(
    forward_graph=fwd_g,
    loss_reduction=LossReduction.MEAN,
    observed_shapes={"y": [2, 4]},
    dtype="float32",
  )
  assert isinstance(bwd_g, LogicalGraph)
  assert any("grad" in nid for nid in bwd_g.nodes)


def test_build_hydrated_forward_graph_fallbacks() -> None:
  """Verify build_hydrated_forward_graph fallback when code has no graph nodes or ast_to_ir fails."""
  empty_code = """pass
"""
  tree = cst.parse_module(empty_code)
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  with mock.patch("importlib.import_module", side_effect=ImportError("no compiler")):
    fwd_g, _ = build_hydrated_forward_graph(
      code=empty_code,
      tree=tree,
      input_shapes=[(2, 4)],
      output_shape=(2, 4),
      dtype="float32",
      node_telemetry={},
      observed_shapes={},
      config=cfg,
    )
    assert "x" in fwd_g.nodes
    assert "out" in fwd_g.nodes

    bwd_g = synthesize_backward_vjp_graph(
      forward_graph=fwd_g,
      loss_reduction=LossReduction.SUM,
      observed_shapes={},
      dtype="float32",
    )
    assert "grad_loss" in bwd_g.nodes


def test_synthesize_backward_vjp_graph_with_output_and_parameter_nodes() -> None:
  """Verify synthesize_backward_vjp_graph handles explicit Output and Parameter nodes."""
  g = LogicalGraph(name="ExplicitOutNet")
  g.nodes["in_0"] = LogicalNode(id="in_0", op_type="Placeholder", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["w_0"] = LogicalNode(id="w_0", op_type="Parameter", shape_metadata=(4, 4), dtype=DType.float32)
  g.nodes["lin"] = LogicalNode(id="lin", op_type="Linear", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["output"] = LogicalNode(id="output", op_type="Output", shape_metadata=(2, 4), dtype=DType.float32)
  g.edges.extend(
    [
      LogicalEdge(source="in_0", target="lin"),
      LogicalEdge(source="w_0", target="lin"),
      LogicalEdge(source="lin", target="output"),
    ]
  )
  g.outputs = ["output"]

  bwd_g = synthesize_backward_vjp_graph(
    forward_graph=g,
    loss_reduction=LossReduction.SUM,
    observed_shapes={},
    dtype="float32",
  )
  assert isinstance(bwd_g, LogicalGraph)
  assert len(bwd_g.nodes) >= len(g.nodes)


def test_run_shared_ir_bridge_across_multiple_backends() -> None:
  """Verify run_shared_ir_bridge emits kernels for webgpu, wasm_simd, webgl, and cuda."""
  code = """import torch
def f(x):
    return torch.relu(x)
"""
  for backend in (
    ExecutionBackend.WEBGPU,
    ExecutionBackend.WASM_SIMD,
    ExecutionBackend.WEBGL,
    ExecutionBackend.CUDA,
  ):
    cfg = RuntimeConfig(
      source_framework="torch",
      target_framework="jax",
      enable_on_device_execution=True,
      execution_backend=backend,
      observed_runtime_shapes={"x": [2, 4]},
    )
    report = run_on_device_trace(code, source_fw="torch", target_fw="jax", config=cfg)
    assert report.executed is True
    assert report.forward_kernel_code is not None
    assert report.backward_kernel_code is not None
    assert len(report.forward_ir_python) > 0
    assert len(report.backward_ir_python) > 0
    assert len(report.forward_mermaid) > 0
    assert len(report.backward_mermaid) > 0

  class ZeroArgGen:
    """Mock generator accepting graph in generate(graph) instead of __init__."""

    def generate(self, *args: LogicalGraph) -> str:
      """Return kernel string.

      Args:
          *args: Graph object.

      Returns:
          Kernel string.
      """
      return "zero_arg_kernel"

  dummy_gen_mod = types.ModuleType("dummy_gen")
  setattr(dummy_gen_mod, "ZeroArgGen", ZeroArgGen)
  assert _invoke_codegen(dummy_gen_mod, "ZeroArgGen", LogicalGraph()) == "zero_arg_kernel"
  assert _resolve_ir_dtype(DType.float16) == DType.float16
  assert _resolve_ir_dtype("int32") == DType.int32
  assert _resolve_ir_dtype("unknown_dtype") == DType.float32
