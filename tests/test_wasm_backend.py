"""Test suite for the wasm backend."""

from ml_switcheroo.core.compiler.backends.wasm_backend import WasmBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from typing import List


def test_wasm_backend_init() -> None:
  """Test element."""
  backend: WasmBackend = WasmBackend(semantics="dummy")
  assert getattr(backend, "semantics") == "dummy"


def test_wasm_backend_compile() -> None:
  """Test element."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="in2", kind="Input"),
    LogicalNode(id="add1", kind="Add"),
    LogicalNode(id="mul1", kind="Mul"),
    LogicalNode(id="sub1", kind="Sub"),
    LogicalNode(id="other1", kind="Other"),
    LogicalNode(id="out1", kind="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="in1", target="add1"),
    LogicalEdge(source="in2", target="add1"),
    LogicalEdge(source="add1", target="mul1"),
    LogicalEdge(source="in2", target="mul1"),
    LogicalEdge(source="add1", target="sub1"),
    LogicalEdge(source="in1", target="sub1"),
    LogicalEdge(source="sub1", target="other1"),
    LogicalEdge(source="other1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(name="test_graph", nodes=nodes, edges=edges)
  backend: WasmBackend = WasmBackend()
  code: str = backend.compile(graph)

  assert "(module" in code
  assert "(func $test_graph" in code
  assert '(export "test_graph")' in code
  assert "(param $arg0 f32)" in code
  assert "(param $arg1 f32)" in code
  assert "(result f32)" in code

  assert "(local $add1 f32)" in code
  assert "f32.add" in code
  assert "f32.mul" in code
  assert "f32.sub" in code
  assert "call $Other" in code
  assert "local.set $add1" in code
  assert "local.set $other1" in code


def test_wasm_backend_compile_no_output() -> None:
  """Test element."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="add1", kind="Add"),
  ]
  edges: List[LogicalEdge] = [LogicalEdge(source="in1", target="add1")]
  graph: LogicalGraph = LogicalGraph(name="main", nodes=nodes, edges=edges)
  backend: WasmBackend = WasmBackend()
  code: str = backend.compile(graph)

  assert "(func $main" in code
  assert "(result f32)" not in code
