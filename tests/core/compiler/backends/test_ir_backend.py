"""Unit tests for the IR Compiler Backend."""

import json
from typing import Any, Dict
import pytest

from ml_switcheroo.core.compiler.backends.ir import IrBackend
from ml_switcheroo.core.compiler.ir import (
  LogicalEdge,
  LogicalGraph,
  LogicalMesh,
  LogicalNode,
  PartitionSpec,
)


def test_ir_backend_compile_json_basic() -> None:
  """Test basic graph compilation to JSON."""
  nodes = {
    "x": LogicalNode(id="x", op_type="Input"),
    "relu1": LogicalNode(id="relu1", op_type="Relu", attributes={"approximate": "none"}),
  }
  edges = [LogicalEdge(source="x", target="relu1")]
  graph = LogicalGraph(name="SimpleNet", nodes=nodes, edges=edges)

  backend = IrBackend(format="json")
  output_json = backend.compile(graph)

  data: Dict[str, Any] = json.loads(output_json)
  assert data["name"] == "SimpleNet"
  assert len(data["nodes"]) == 2
  assert len(data["edges"]) == 1
  assert data["edges"][0] == {"source": "x", "target": "relu1"}

  relu_node = next(n for n in data["nodes"] if n["id"] == "relu1")
  assert relu_node["op_type"] == "Relu"
  assert relu_node["kind"] == "Relu"
  assert relu_node["attributes"]["approximate"] == "none"
  assert relu_node["metadata"]["approximate"] == "none"
  assert relu_node["inputs"] == ["x"]


def test_ir_backend_compile_json_sharding_and_mesh() -> None:
  """Test graph compilation preserving mesh and sharding PartitionSpec."""
  mesh = LogicalMesh(shape={"data": 2, "model": 4})
  nodes = {
    "conv1": LogicalNode(
      id="conv1",
      op_type="Conv",
      attributes={"kernel_size": "3"},
      sharding=PartitionSpec(axes=("data", None, "model")),
    )
  }
  graph = LogicalGraph(name="ShardedNet", nodes=nodes, edges=[], mesh=mesh)

  backend = IrBackend()
  output_json = backend.compile(graph)

  data: Dict[str, Any] = json.loads(output_json)
  assert "mesh" in data
  assert data["mesh"]["shape"] == {"data": 2, "model": 4}

  conv_node = data["nodes"][0]
  assert "sharding" in conv_node
  assert conv_node["sharding"]["axes"] == ["data", None, "model"]


def test_ir_backend_compile_json_determinism() -> None:
  """Test JSON determinism with identical sorted output across multiple calls."""
  nodes = {
    "b": LogicalNode(id="b", op_type="Add", attributes={"z": "1", "a": "2"}),
    "a": LogicalNode(id="a", op_type="Input"),
  }
  edges = [LogicalEdge(source="a", target="b")]
  graph = LogicalGraph(name="DetNet", nodes=nodes, edges=edges)

  backend = IrBackend()
  out1 = backend.compile(graph)
  out2 = backend.compile(graph)
  assert out1 == out2


def test_ir_backend_compile_python() -> None:
  """Test compilation to executable Python code creating LogicalGraph."""
  mesh = LogicalMesh(shape={"device": 8})
  nodes = {
    "in_x": LogicalNode(
      id="in_x",
      op_type="Input",
      sharding=PartitionSpec(axes=("device",)),
    ),
    "fc": LogicalNode(id="fc", op_type="Gemm"),
  }
  edges = [LogicalEdge(source="in_x", target="fc")]
  graph = LogicalGraph(name="PyNet", nodes=nodes, edges=edges, mesh=mesh)

  backend = IrBackend(format="python")
  code = backend.compile(graph)

  assert "import ml_switcheroo_ir as sw_ir" in code
  assert "def build_graph() -> sw_ir.LogicalGraph:" in code
  assert "LogicalMesh(shape={'device': 8})" in code
  assert "PartitionSpec(axes=('device',))" in code
  assert "nodes = {" in code
  assert "op_type='Input'" in code
  assert "op_type='Gemm'" in code
  assert "LogicalGraph(" in code
  assert "name='PyNet'" in code

  # Also test without mesh
  graph_no_mesh = LogicalGraph(name="NoMesh", nodes=nodes, edges=edges, mesh=None)
  code_no_mesh = backend.compile(graph_no_mesh)
  assert "mesh = None" in code_no_mesh


def test_ir_backend_validation_invalid_input() -> None:
  """Test validation error when passing an invalid object to compile."""
  backend = IrBackend()
  with pytest.raises(ValueError, match="instance of LogicalGraph"):
    backend.compile("not a graph")  # type: ignore


def test_ir_backend_validation_dangling_edges() -> None:
  """Test validation errors for dangling source or target references in edges."""
  backend = IrBackend()

  # Dangling source
  graph_bad_src = LogicalGraph(
    name="BadSrc",
    nodes={"target_node": LogicalNode(id="target_node", op_type="Output")},
    edges=[LogicalEdge(source="missing_source", target="target_node")],
  )
  with pytest.raises(ValueError, match="Dangling edge source"):
    backend.compile(graph_bad_src)

  # Dangling target
  graph_bad_tgt = LogicalGraph(
    name="BadTgt",
    nodes={"src_node": LogicalNode(id="src_node", op_type="Input")},
    edges=[LogicalEdge(source="src_node", target="missing_target")],
  )
  with pytest.raises(ValueError, match="Dangling edge target"):
    backend.compile(graph_bad_tgt)
