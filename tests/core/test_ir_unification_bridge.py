"""Unit tests for Phase 10: Zero-Copy `ml-switcheroo-ir` (`LogicalGraph`) Bridge."""

from __future__ import annotations

import types
from unittest import mock

import libcst as cst
import pytest
from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.config import (
  ConversionMode,
  ExecutionBackend,
  LossReduction,
  OptimizationLevel,
  RuntimeConfig,
)
from ml_switcheroo.core.ir_bridge import (
  ForwardGraphBuildRequest,
  _resolve_ir_dtype,
  build_hydrated_forward_graph,
  synthesize_backward_vjp_graph,
)
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry
from ml_switcheroo.core.runtime_tracer import _invoke_codegen, run_on_device_trace


def test_build_hydrated_forward_graph_and_autodiff_grad(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify build_hydrated_forward_graph and synthesize_backward_vjp_graph on shared LogicalGraph.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
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
  stub_ast_to_ir = types.ModuleType("ml_switcheroo_compiler.backends.ast_to_ir")
  parsed_g = LogicalGraph(name="ParsedNet")
  parsed_g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  parsed_g.nodes["y"] = LogicalNode(id="y", op_type="Add", dtype=DType.float32)
  parsed_g.edges.append(LogicalEdge(source="x", target="y"))
  monkeypatch.setattr(stub_ast_to_ir, "parse_ast_to_ir", lambda _code: parsed_g, raising=False)

  stub_shape_inf = types.ModuleType("ml_switcheroo_compiler.transforms.passes.shape_inference")
  monkeypatch.setattr(stub_shape_inf, "shape_inference_pass", lambda _g: None, raising=False)
  monkeypatch.setattr(stub_shape_inf, "annotate_learned_shapes", lambda _g, _s: None, raising=False)

  stub_autodiff = types.ModuleType("ml_switcheroo_compiler.transforms.autodiff")

  def _stub_grad(graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
    """Return a backward LogicalGraph with grad_x node.

    Args:
        graph: Forward graph.
        wrt: WRT node IDs.
        output_id: Loss node ID.

    Returns:
        Backward graph.
    """
    bg = graph.model_copy(deep=True)
    bg.nodes["grad_x"] = LogicalNode(id="grad_x", op_type="Mul", shape_metadata=(2, 4), dtype=DType.float32)
    return bg

  monkeypatch.setattr(stub_autodiff, "grad", _stub_grad, raising=False)

  def _stub_import(name: str) -> types.ModuleType:
    """Import stub for ast_to_ir, shape_inference, and autodiff.

    Args:
        name: Module name.

    Returns:
        Stubbed module.
    """
    if name == "ml_switcheroo_compiler.backends.ast_to_ir":
      return stub_ast_to_ir
    if name == "ml_switcheroo_compiler.transforms.passes.shape_inference":
      return stub_shape_inf
    if name == "ml_switcheroo_compiler.transforms.autodiff":
      return stub_autodiff
    raise ImportError(name)

  with mock.patch("importlib.import_module", side_effect=_stub_import):
    fwd_g, passes = build_hydrated_forward_graph(
      ForwardGraphBuildRequest(
        code=code,
        tree=tree,
        input_shapes=[(2, 4)],
        output_shape=(2, 4),
        dtype="float32",
        config=cfg,
        node_telemetry=tel,
        observed_shapes={"y": [2, 4]},
      )
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

  # Also test when parse_ast_to_ir returns an empty LogicalGraph (hits 298->302 branch)
  # and when parse_ast_to_ir returns a graph with no Input/Placeholder nodes (hits 334-335)
  stub_no_in_ast_to_ir = types.ModuleType("ml_switcheroo_compiler.backends.ast_to_ir")
  no_in_g = LogicalGraph(name="NoInputNet")
  no_in_g.nodes["op1"] = LogicalNode(id="op1", op_type="Relu", dtype=DType.float32)
  monkeypatch.setattr(stub_no_in_ast_to_ir, "parse_ast_to_ir", lambda _code: no_in_g, raising=False)
  with mock.patch(
    "importlib.import_module",
    side_effect=lambda name: (
      stub_no_in_ast_to_ir
      if name == "ml_switcheroo_compiler.backends.ast_to_ir"
      else (_ for _ in ()).throw(ImportError(name))
    ),
  ):
    fwd_no_in, _ = build_hydrated_forward_graph(
      ForwardGraphBuildRequest(
        code=code,
        tree=tree,
        input_shapes=[(2, 4)],
        output_shape=(2, 4),
        dtype="float32",
        config=cfg,
        node_telemetry=tel,
        observed_shapes={"y": [2, 4]},
      )
    )
    assert fwd_no_in.inputs == ["op1"]

  stub_empty_ast_to_ir = types.ModuleType("ml_switcheroo_compiler.backends.ast_to_ir")
  monkeypatch.setattr(stub_empty_ast_to_ir, "parse_ast_to_ir", lambda _code: LogicalGraph(), raising=False)
  with mock.patch(
    "importlib.import_module",
    side_effect=lambda name: (
      stub_empty_ast_to_ir
      if name == "ml_switcheroo_compiler.backends.ast_to_ir"
      else (_ for _ in ()).throw(ImportError(name))
    ),
  ):
    fwd_g2, _ = build_hydrated_forward_graph(
      ForwardGraphBuildRequest(
        code=code,
        tree=tree,
        input_shapes=[(2, 4)],
        output_shape=(2, 4),
        dtype="float32",
        config=cfg,
        node_telemetry=tel,
        observed_shapes={"y": [2, 4]},
      )
    )
    assert len(fwd_g2.nodes) >= 1


def test_build_hydrated_forward_graph_fallbacks() -> None:
  """Verify build_hydrated_forward_graph fallback when code has no graph nodes or ast_to_ir fails."""
  empty_code = """pass
"""
  tree = cst.parse_module(empty_code)
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  with mock.patch("importlib.import_module", side_effect=ImportError("no compiler")):
    fwd_g, _ = build_hydrated_forward_graph(
      ForwardGraphBuildRequest(
        code=empty_code,
        tree=tree,
        input_shapes=[(2, 4)],
        output_shape=(2, 4),
        dtype="float32",
        config=cfg,
      )
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


def test_run_shared_ir_bridge_across_multiple_backends(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify run_shared_ir_bridge emits kernels for webgpu, wasm_simd, webgl, and cuda.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
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
  monkeypatch.setattr(dummy_gen_mod, "ZeroArgGen", ZeroArgGen, raising=False)
  assert _invoke_codegen(dummy_gen_mod, "ZeroArgGen", LogicalGraph()) == "zero_arg_kernel"
  assert _resolve_ir_dtype(DType.float16) == DType.float16
  assert _resolve_ir_dtype("int32") == DType.int32
  assert _resolve_ir_dtype("unknown_dtype") == DType.float32
