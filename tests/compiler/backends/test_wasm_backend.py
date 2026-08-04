"""Test suite for the WASM Compiler Backend."""

from ml_switcheroo.core.compiler.backends.wasm_backend import WasmBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


def test_wasm_backend_compile() -> None:
  """Tests WASM compilation."""
  graph = LogicalGraph(name="test_graph")
  graph.nodes.append(LogicalNode(id="in1", kind="Input"))
  graph.nodes.append(LogicalNode(id="in2", kind="Input"))
  graph.nodes.append(LogicalNode(id="add1", kind="Add"))
  graph.nodes.append(LogicalNode(id="mul1", kind="Mul"))
  graph.nodes.append(LogicalNode(id="out1", kind="Output"))

  graph.edges.append(LogicalEdge(source="in1", target="add1"))
  graph.edges.append(LogicalEdge(source="in2", target="add1"))

  graph.edges.append(LogicalEdge(source="add1", target="mul1"))
  graph.edges.append(LogicalEdge(source="in1", target="mul1"))

  graph.edges.append(LogicalEdge(source="mul1", target="out1"))

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
  """Tests WASM compilation with Sub and custom nodes."""
  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="n1", kind="Sub"))
  graph.nodes.append(LogicalNode(id="n2", kind="MyCustomOp"))
  graph.edges.append(LogicalEdge(source="n1", target="n2"))

  backend = WasmBackend()
  wat = backend.compile(graph)

  assert "f32.sub" in wat
  assert "call $MyCustomOp" in wat
  assert "(func $Model" in wat


def test_wasm_backend_output_no_incoming() -> None:
  """Tests output node with no incoming edges."""
  graph = LogicalGraph(name="")
  graph.nodes.append(LogicalNode(id="out1", kind="Output"))
  backend = WasmBackend()
  wat = backend.compile(graph)
  assert '(func $main (export "main")' in wat
  assert "(result f32)" in wat


def test_wasm_backend_incoming_not_found() -> None:
  """Tests incoming node that is not in the graph."""
  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="n1", kind="Add"))
  graph.edges.append(LogicalEdge(source="missing", target="n1"))
  backend = WasmBackend()
  wat = backend.compile(graph)
  assert "local.get $missing" in wat
