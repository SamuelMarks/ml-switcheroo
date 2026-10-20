"""Test suite for the wasm backend."""

import typing

from ml_switcheroo.core.compiler.backends.wasm_backend import WasmBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_wasm_backend_init() -> None:
  """Verify wasm backend init behavior."""
  backend: WasmBackend = WasmBackend(semantics="dummy")  # type: ignore[arg-type]
  assert getattr(backend, "semantics") == "dummy"


def test_wasm_backend_compile() -> None:
  """Verify wasm backend compile behavior."""
  nodes: typing.List[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="in2", op_type="Input"),
    LogicalNode(id="add1", op_type="Add"),
    LogicalNode(id="mul1", op_type="Mul"),
    LogicalNode(id="sub1", op_type="Sub"),
    LogicalNode(id="other1", op_type="Other"),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: typing.List[LogicalEdge] = [
    LogicalEdge(source="in1", target="add1"),
    LogicalEdge(source="in2", target="add1"),
    LogicalEdge(source="add1", target="mul1"),
    LogicalEdge(source="in2", target="mul1"),
    LogicalEdge(source="add1", target="sub1"),
    LogicalEdge(source="in1", target="sub1"),
    LogicalEdge(source="sub1", target="other1"),
    LogicalEdge(source="other1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(name="test_graph", nodes={n.id: n for n in nodes}, edges=edges)
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
  """Verify wasm backend compile no output behavior."""
  nodes: typing.List[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="add1", op_type="Add"),
  ]
  edges: typing.List[LogicalEdge] = [LogicalEdge(source="in1", target="add1")]
  graph: LogicalGraph = LogicalGraph(name="main", nodes={n.id: n for n in nodes}, edges=edges)
  backend: WasmBackend = WasmBackend()
  code: str = backend.compile(graph)

  assert "(func $main" in code
  assert "(result f32)" not in code
