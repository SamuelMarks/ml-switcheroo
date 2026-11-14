"""Unit tests for Phase 11: Live `zero-*` `ProxyTensor` Tracing & `evaluate_graph`."""

from __future__ import annotations

import types
from unittest import mock

import pytest
from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.ir_bridge import (
  evaluate_shared_backward_graph,
  trace_with_zero_proxy_tensors,
)
from ml_switcheroo.core.runtime_tracer import run_on_device_trace


class _StubTracingState:
  """Deterministic stub for `ml_switcheroo_compiler.tracing.global_tracing_state`."""

  def __init__(self) -> None:
    """Initialize stub tracing state."""
    self.nodes: dict[str, LogicalNode] = {}

  def start_tracing(self) -> None:
    """Reset recorded nodes at trace start."""
    self.nodes = {}

  def add_node(self, node: LogicalNode) -> None:
    """Record a LogicalNode in the active trace.

    Args:
        node: LogicalNode to record.
    """
    self.nodes[node.id] = node

  def stop_tracing(self) -> LogicalGraph:
    """Return a LogicalGraph populated with recorded nodes.

    Returns:
        Populated `LogicalGraph`.
    """
    return LogicalGraph(nodes=dict(self.nodes))


class _StubProxyTensor:
  """Deterministic stub for `ml_switcheroo_compiler.tracing.ProxyTensor`."""

  def __init__(self, id: str, shape: list[int], dtype: str = "float32") -> None:
    """Initialize stub ProxyTensor.

    Args:
        id: Node identifier.
        shape: Tensor shape list.
        dtype: Data type string.
    """
    self.id = id
    self.shape = shape
    self.dtype = dtype

  def __add__(self, other: "_StubProxyTensor | int | float") -> "_StubProxyTensor":
    """Support binary addition during traced execution.

    Args:
        other: RHS operand.

    Returns:
        Self stub proxy tensor.
    """
    return self


def _make_stub_tracing_mod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
  """Build a stub `ml_switcheroo_compiler.tracing` module satisfying `_TracingModuleProtocol`.

  Args:
      monkeypatch: Pytest monkeypatch fixture for scoped module attribute setup.

  Returns:
      Configured `types.ModuleType` instance.
  """
  mod = types.ModuleType("ml_switcheroo_compiler.tracing")
  monkeypatch.setattr(mod, "global_tracing_state", _StubTracingState(), raising=False)
  monkeypatch.setattr(mod, "ProxyTensor", _StubProxyTensor, raising=False)
  return mod


def test_trace_with_zero_proxy_tensors_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify trace_with_zero_proxy_tensors captures a LogicalGraph via global_tracing_state.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = """_helper = lambda z: z
def _private_fn(x):
    return x
def f(x, y):
    return x + y
"""
  stub_mod = _make_stub_tracing_mod(monkeypatch)
  with mock.patch("importlib.import_module", return_value=stub_mod):
    g = trace_with_zero_proxy_tensors(code, input_shapes=[(2, 4), (2, 4)], dtype="float32", seed=42)
  assert isinstance(g, LogicalGraph)
  assert "x" in g.nodes
  assert "input_1" in g.nodes


def test_trace_with_zero_proxy_tensors_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify trace_with_zero_proxy_tensors handles import errors, missing attrs, and empty graphs.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  with mock.patch("importlib.import_module", side_effect=ImportError("no tracing")):
    assert trace_with_zero_proxy_tensors("def f(x): return x", [(2, 4)]) is None

  empty_mod = types.ModuleType("empty_mod")
  with mock.patch("importlib.import_module", return_value=empty_mod):
    assert trace_with_zero_proxy_tensors("def f(x): return x", [(2, 4)]) is None

  stub_mod = _make_stub_tracing_mod(monkeypatch)
  with mock.patch("importlib.import_module", return_value=stub_mod):
    # Code that raises inside callable
    bad_code = """def f(x):
    raise ValueError("fail")
"""
    g_bad = trace_with_zero_proxy_tensors(bad_code, input_shapes=[(2, 4)])
    assert g_bad is not None

    # Code with no public top-level function
    no_fn_code = """def _hidden(x):
    return x
"""
    g_no_fn = trace_with_zero_proxy_tensors(no_fn_code, input_shapes=[(2, 4)])
    assert g_no_fn is not None

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

  monkeypatch.setattr(mock_state_mod, "global_tracing_state", EmptyState(), raising=False)
  monkeypatch.setattr(mock_state_mod, "ProxyTensor", lambda **kwargs: kwargs, raising=False)
  monkeypatch.setattr(mock_state_mod, "Tensor", lambda *args, **kwargs: args, raising=False)
  with mock.patch("importlib.import_module", return_value=mock_state_mod):
    assert trace_with_zero_proxy_tensors("x = 1", [(2, 4)]) is None


def test_evaluate_shared_backward_graph_live_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify evaluate_shared_backward_graph evaluates nodes and falls back cleanly on error.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  g = LogicalGraph(name="EvalGraph")
  g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["empty_shape"] = LogicalNode(id="empty_shape", op_type="Placeholder", shape_metadata=(), dtype=DType.float32)
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

  eval_mod = types.ModuleType("ml_switcheroo_compiler.interpreter.evaluator")
  tensor_mod = types.ModuleType("ml_switcheroo_compiler.core.tensor")

  class _StubShapeObj:
    """Stub evaluated tensor exposing shape."""

    def __init__(self, shape: tuple[int, ...]) -> None:
      """Initialize stub shape object.

      Args:
          shape: Shape tuple.
      """
      self.shape = shape

  monkeypatch.setattr(
    eval_mod,
    "evaluate_graph",
    lambda graph, env: {"x": _StubShapeObj((2, 4)), "y": _StubShapeObj((2, 4))},
    raising=False,
  )
  monkeypatch.setattr(tensor_mod, "TensorConfig", lambda shape, dtype: (shape, dtype), raising=False)
  monkeypatch.setattr(tensor_mod, "Tensor", lambda data, config: (data, config), raising=False)

  def _stub_import(name: str) -> types.ModuleType:
    """Import stub for evaluator and tensor modules.

    Args:
        name: Module path.

    Returns:
        Stubbed module.
    """
    if name == "ml_switcheroo_compiler.interpreter.evaluator":
      return eval_mod
    if name == "ml_switcheroo_compiler.core.tensor":
      return tensor_mod
    raise ImportError(name)

  with mock.patch("importlib.import_module", side_effect=_stub_import):
    shapes = evaluate_shared_backward_graph(g, input_shapes=[(2, 4)], dtype="float32", seed=7)
  assert shapes["x"] == [2, 4]
  assert shapes["y"] == [2, 4]
  assert shapes["empty_shape"] == []

  with mock.patch("importlib.import_module", side_effect=ImportError("no evaluator")):
    fallback_shapes = evaluate_shared_backward_graph(g, input_shapes=[], dtype="float32", seed=7)
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
