"""Test suite for the WASM Compiler Backend."""

from ml_switcheroo.core.compiler.backends.wasm_backend import WasmBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_wasm_backend_compile() -> None:
  """Docstring."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "in2": LogicalNode(id="in2", op_type="Input"),
    "add1": LogicalNode(id="add1", op_type="Add"),
    "mul1": LogicalNode(id="mul1", op_type="Mul"),
    "out1": LogicalNode(id="out1", op_type="Output"),
  }
  edges = [
    LogicalEdge(source="in1", target="add1"),
    LogicalEdge(source="in2", target="add1"),
    LogicalEdge(source="add1", target="mul1"),
    LogicalEdge(source="in1", target="mul1"),
    LogicalEdge(source="mul1", target="out1"),
  ]
  graph = LogicalGraph(name="test_graph", nodes=nodes, edges=edges)

  backend = WasmBackend(semantics="dummy_semantics")
  assert backend.semantics == "dummy_semantics"
  wat = backend.compile(graph)

  assert '(func $test_graph (export "test_graph")' in wat
  assert "(param $arg0 f32) (param $arg1 f32)" in wat
  assert "(result f32)" in wat
  assert "(local $add1 f32)" in wat
  assert "local.get $arg0" in wat
  assert "local.get $arg1" in wat
  assert "f32.add" in wat
  assert "local.set $add1" in wat
  assert "local.get $add1" in wat
  assert "f32.mul" in wat
  assert "local.set $mul1" in wat
  assert "local.get $mul1" in wat


def test_wasm_backend_sub_and_custom() -> None:
  """Docstring."""
  nodes = {
    "n1": LogicalNode(id="n1", op_type="Sub"),
    "n2": LogicalNode(id="n2", op_type="MyCustomOp"),
  }
  edges = [LogicalEdge(source="n1", target="n2")]
  graph = LogicalGraph(nodes=nodes, edges=edges)

  backend = WasmBackend()
  wat = backend.compile(graph)

  assert "f32.sub" in wat
  assert "call $MyCustomOp" in wat
  assert "(func $Model" in wat


def test_wasm_backend_output_no_incoming() -> None:
  """Docstring."""
  nodes = {"out1": LogicalNode(id="out1", op_type="Output")}
  graph = LogicalGraph(name="", nodes=nodes)
  backend = WasmBackend()
  wat = backend.compile(graph)
  assert '(func $main (export "main")' in wat
  assert "(result f32)" in wat


def test_wasm_backend_incoming_not_found() -> None:
  """Docstring."""
  nodes = {"n1": LogicalNode(id="n1", op_type="Add")}
  edges = [LogicalEdge(source="missing", target="n1")]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  backend = WasmBackend()
  wat = backend.compile(graph)
  assert "local.get $missing" in wat
