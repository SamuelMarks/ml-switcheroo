"""Unit tests for Phase 11: Live `zero-*` `ProxyTensor` Tracing & `evaluate_graph`."""

from __future__ import annotations
from tests.conftest import _test_setattr

import types
from unittest import mock

from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.ir_bridge import (
  evaluate_shared_backward_graph,
  trace_with_zero_proxy_tensors,
)
from ml_switcheroo.core.runtime_tracer import run_on_device_trace


def test_trace_with_zero_proxy_tensors_live() -> None:
  """Verify trace_with_zero_proxy_tensors captures a LogicalGraph via global_tracing_state."""
  code = """def f(x, y):
    return x + y
"""
  g = trace_with_zero_proxy_tensors(code, input_shapes=[(2, 4), (2, 4)], dtype="float32", seed=42)
  assert isinstance(g, LogicalGraph)
  assert "x" in g.nodes
  assert "input_1" in g.nodes


def test_trace_with_zero_proxy_tensors_edge_cases() -> None:
  """Verify trace_with_zero_proxy_tensors handles import errors, missing attrs, and empty graphs."""
  with mock.patch("importlib.import_module", side_effect=ImportError("no tracing")):
    assert trace_with_zero_proxy_tensors("def f(x): return x", [(2, 4)]) is None

  empty_mod = types.ModuleType("empty_mod")
  with mock.patch("importlib.import_module", return_value=empty_mod):
    assert trace_with_zero_proxy_tensors("def f(x): return x", [(2, 4)]) is None

  # Code that raises inside exec or callable
  bad_code = """def f(x):
    raise ValueError("fail")
"""
  g_bad = trace_with_zero_proxy_tensors(bad_code, input_shapes=[(2, 4)])
  assert g_bad is not None

  syntax_err_code = """def f(x):
    1 / 0
f(1)
"""
  g_exec_err = trace_with_zero_proxy_tensors(syntax_err_code, input_shapes=[(2, 4)])
  assert g_exec_err is not None

  # When stop_tracing returns an empty LogicalGraph, returns None
  mock_state_mod = types.ModuleType("state_mod")

  class EmptyState:
    """Mock tracing state returning an empty graph."""

    def start_tracing(self) -> None:
      """Start tracing."""

    def add_node(self, node: LogicalNode) -> None:
      """Ignore node.

      Args:
          node: LogicalNode.
      """

    def stop_tracing(self) -> LogicalGraph:
      """Return empty LogicalGraph.

      Returns:
          Empty `LogicalGraph`.
      """
      return LogicalGraph()

  _test_setattr(mock_state_mod, "global_tracing_state", EmptyState())
  _test_setattr(mock_state_mod, "ProxyTensor", lambda **kwargs: kwargs)
  _test_setattr(mock_state_mod, "Tensor", lambda *args, **kwargs: args)
  with mock.patch("importlib.import_module", return_value=mock_state_mod):
    assert trace_with_zero_proxy_tensors("x = 1", [(2, 4)]) is None


def test_evaluate_shared_backward_graph_live_and_fallback() -> None:
  """Verify evaluate_shared_backward_graph evaluates nodes and falls back cleanly on error."""
  g = LogicalGraph(name="EvalGraph")
  g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["y"] = LogicalNode(id="y", op_type="Add", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["no_meta"] = LogicalNode(id="no_meta", op_type="Parameter", dtype=DType.float32)
  g.edges.extend(
    [
      LogicalEdge(source="x", target="y"),
      LogicalEdge(source="x", target="y"),
    ]
  )
  g.inputs = ["x"]
  g.outputs = ["y"]

  shapes = evaluate_shared_backward_graph(g, input_shapes=[(2, 4)], dtype="float32", seed=7)
  assert shapes["x"] == [2, 4]
  assert shapes["y"] == [2, 4]

  with mock.patch("importlib.import_module", side_effect=ImportError("no evaluator")):
    fallback_shapes = evaluate_shared_backward_graph(g, input_shapes=[(2, 4)], dtype="float32", seed=7)
    assert fallback_shapes["x"] == [2, 4]


def test_zero_frameworks_live_trace_execution() -> None:
  """Verify run_on_device_trace executes across torch, jax, flax_nnx, keras, tensorflow, mlx, and paxml."""
  code = """def forward(x):
    y = x * 2.0 + 1.0
    return y
"""
  for fw in ("torch", "jax", "flax_nnx", "keras", "tensorflow", "mlx", "paxml"):
    cfg = RuntimeConfig(
      source_framework=fw,
      target_framework="jax" if fw != "jax" else "torch",
      enable_on_device_execution=True,
    )
    report = run_on_device_trace(code, source_fw=fw, target_fw=cfg.target_framework, config=cfg)
    assert report.executed is True
    assert report.output_shape == [2, 4]
    assert report.finite_check_passed is True
