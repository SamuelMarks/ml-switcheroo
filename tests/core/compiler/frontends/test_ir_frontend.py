"""Unit tests for the IR Compiler Frontend."""

import json
from typing import Any, Dict
import libcst as cst
import pytest

from ml_switcheroo.core.compiler.frontends.ir import (
  IrFrontend,
  IrJsonParser,
  IrLifter,
  IrParseError,
  IrPythonParser,
  IrToCstGenerator,
)
from ml_switcheroo.core.compiler.ir import (
  LogicalEdge,
  LogicalGraph,
  LogicalNode,
)


def test_ir_frontend_parse_valid_json() -> None:
  """Test parsing valid JSON with nodes, edges, sharding, and mesh."""
  payload: Dict[str, Any] = {
    "name": "ConvNet",
    "mesh": {"shape": {"data": 2, "model": 4}},
    "nodes": [
      {
        "id": "in1",
        "op_type": "Input",
        "kind": "Input",
        "domain": "ai.onnx",
        "version": 17,
        "attributes": {"dtype": "float32"},
        "metadata": {"dtype": "float32"},
        "sharding": {"axes": ["data", None]},
      },
      {
        "id": "conv1",
        "op_type": "Conv",
        "attributes": {"kernel_size": "3"},
        "metadata": {"kernel_size": "3"},
        "inputs": ["in1"],
      },
    ],
    "edges": [{"source": "in1", "target": "conv1"}],
  }
  json_str = json.dumps(payload)

  frontend = IrFrontend(json_str)
  graph = frontend.parse_to_graph()

  assert graph.name == "ConvNet"
  assert len(graph.nodes) == 2
  assert len(graph.edges) == 1
  assert graph.mesh is not None
  assert graph.mesh.shape == {"data": 2, "model": 4}

  node_map = dict(graph.nodes)
  assert "in1" in node_map
  assert "conv1" in node_map
  assert node_map["conv1"].op_type == "Conv"
  assert node_map["in1"].sharding is not None
  assert node_map["in1"].sharding.axes == ("data", None)


def test_ir_frontend_parse_json_dict_nodes() -> None:
  """Test parsing JSON when nodes is formatted as a dictionary instead of a list."""
  payload: Dict[str, Any] = {
    "name": "DictNet",
    "nodes": {
      "a": {"id": "a", "op_type": "Input", "kind": "Input"},
      "b": {"id": "b", "op_type": "Relu", "kind": "Relu", "inputs": ["a"]},
    },
  }
  json_str = json.dumps(payload)

  parser = IrJsonParser()
  graph = parser.parse(json_str)
  assert len(graph.nodes) == 2
  assert len(graph.edges) == 1
  assert graph.edges[0].source == "a"
  assert graph.edges[0].target == "b"


def test_ir_frontend_parse_malformed_json_syntax() -> None:
  """Test syntax error handling for malformed JSON string."""
  frontend = IrFrontend("{invalid_json:")
  with pytest.raises(IrParseError) as exc_info:
    frontend.parse_to_graph()
  assert exc_info.value.line is not None
  assert "Malformed JSON syntax" in str(exc_info.value)


def test_ir_frontend_parse_non_dict_root() -> None:
  """Test error handling when JSON root is not a dictionary."""
  parser = IrJsonParser()
  with pytest.raises(IrParseError, match="must be a dictionary"):
    parser.parse(json.dumps(["not", "a", "dict"]))


def test_ir_frontend_parse_invalid_nodes_container() -> None:
  """Test error handling when 'nodes' is neither a list nor a dictionary."""
  parser = IrJsonParser()
  with pytest.raises(IrParseError, match="'nodes' field must be a list or dictionary"):
    parser.parse(json.dumps({"nodes": "invalid_string"}))


def test_ir_frontend_parse_node_not_dict() -> None:
  """Test error handling when a node entry is not a dictionary."""
  parser = IrJsonParser()
  with pytest.raises(IrParseError, match="Every node entry must be a dictionary"):
    parser.parse(json.dumps({"nodes": ["not_a_dict"]}))


def test_ir_frontend_parse_node_missing_id() -> None:
  """Test error handling when node misses mandatory 'id' attribute."""
  parser = IrJsonParser()
  with pytest.raises(IrParseError, match="missing mandatory 'id'"):
    parser.parse(json.dumps({"nodes": [{"kind": "Conv"}]}))


def test_ir_frontend_parse_node_missing_kind() -> None:
  """Test error handling when node misses mandatory 'kind' or 'op_type'."""
  parser = IrJsonParser()
  with pytest.raises(IrParseError, match="missing mandatory 'kind'"):
    parser.parse(json.dumps({"nodes": [{"id": "n1"}]}))


def test_ir_frontend_dangling_edges() -> None:
  """Test lifting error when edge points to nonexistent node."""
  lifter = IrLifter()
  bad_src_graph = LogicalGraph(
    name="BadSrc",
    nodes={"n2": LogicalNode(id="n2", op_type="Output")},
    edges=[LogicalEdge(source="n1", target="n2")],
  )
  with pytest.raises(IrParseError, match="Dangling edge source"):
    lifter.lift(bad_src_graph)

  bad_tgt_graph = LogicalGraph(
    name="BadTgt",
    nodes={"n1": LogicalNode(id="n1", op_type="Input")},
    edges=[LogicalEdge(source="n1", target="n2")],
  )
  with pytest.raises(IrParseError, match="Dangling edge target"):
    lifter.lift(bad_tgt_graph)


def test_ir_lifter_duplicate_edges() -> None:
  """Test lifting graph with duplicate edges deduplicates inputs."""
  lifter = IrLifter()
  nodes = {
    "a": LogicalNode(id="a", op_type="Input"),
    "b": LogicalNode(id="b", op_type="Output"),
  }
  edges = [
    LogicalEdge(source="a", target="b"),
  ]
  graph = LogicalGraph(name="DupEdges", nodes=nodes, edges=edges)
  graph._pending_edges = [LogicalEdge(source="a", target="b")]
  lifted = lifter.lift(graph)
  assert len(lifted.nodes) == 2
  assert lifted.nodes["b"].inputs == ["a"]


def test_ir_frontend_parse_python_cst() -> None:
  """Test parsing Python code representing neural modules into a LogicalGraph."""
  code = """
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)
"""
  frontend = IrFrontend(code)
  graph = frontend.parse_to_graph()
  assert graph.name == "MyModel"
  assert len(graph.nodes) >= 1
  assert any(getattr(n, "op_type", None) == "Conv2d" for n in graph.nodes.values())


def test_ir_frontend_parse_python_empty_fallback() -> None:
  """Test Python code with no neural layers returns empty graph."""
  parser = IrPythonParser()
  graph = parser.parse("x = 1 + 2")
  assert len(graph.nodes) == 0


def test_ir_frontend_empty_input() -> None:
  """Test empty or whitespace input returns empty graph."""
  frontend = IrFrontend("   ")
  graph = frontend.parse_to_graph()
  assert len(graph.nodes) == 0


def test_ir_frontend_cst_generator() -> None:
  """Test synthesizing LibCST Module from LogicalGraph."""
  nodes = {
    "x": LogicalNode(id="x", op_type="Input"),
    "conv1": LogicalNode(id="conv1", op_type="Conv2d"),
    "relu1": LogicalNode(id="relu1", op_type="ReLU"),
  }
  edges = [
    LogicalEdge(source="x", target="conv1"),
    LogicalEdge(source="conv1", target="relu1"),
  ]
  graph = LogicalGraph(name="NetToCst", nodes=nodes, edges=edges)

  generator = IrToCstGenerator()
  module = generator.generate(graph)

  assert isinstance(module, cst.Module)
  code = module.code
  assert "class NetToCst(nn.Module):" in code
  assert "def __init__(self):" in code
  assert "def forward(self, x):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "self.relu1 = nn.ReLU()" in code
  assert "return relu1" in code


def test_ir_frontend_cst_generator_empty() -> None:
  """Test synthesizing LibCST Module from empty LogicalGraph."""
  graph = LogicalGraph(name="EmptyNet", nodes={}, edges=[])
  generator = IrToCstGenerator()
  module = generator.generate(graph)
  assert "class EmptyNet(nn.Module):" in module.code
  assert "return x" in module.code


def test_ir_frontend_edge_and_metadata_variations() -> None:
  """Test duplicate edges, non-dict metadata fallback, and edge variations."""
  payload: Dict[str, Any] = {
    "name": "VariationNet",
    "nodes": [
      {
        "id": "x",
        "op_type": "Input",
        "kind": "Input",
        "metadata": "non_dict_metadata",  # triggers fallback
        "inputs": ["external_in", 123],  # 123 is non-string, skipped
      },
      {
        "id": "y",
        "op_type": "Output",
        "kind": "Output",
        "inputs": ["x"],
      },
      {
        "id": "z",
        "op_type": "Output",
        "kind": "Output",
      },
    ],
    "edges": [
      {"source": "x", "target": "y"},  # duplicate of implicit edge from y.inputs
      {"source": "x", "target": "z"},  # non-duplicate explicit edge
      {"invalid": "edge"},  # skipped
      "not_a_dict",  # skipped
    ],
    "mesh": "not_a_dict_mesh",  # skipped
  }
  parser = IrJsonParser()
  graph = parser.parse(json.dumps(payload))
  assert graph.mesh is None
  assert len(graph.edges) == 3  # (external_in -> x), (x -> y), (x -> z)
  assert graph.nodes["x"].attributes == {} or graph.nodes["x"].metadata == {}


def test_ir_frontend_cst_generator_multiple_inputs() -> None:
  """Test synthesizing LibCST Module with a node that has multiple inputs."""
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "in2": LogicalNode(id="in2", op_type="Input"),
    "add": LogicalNode(id="add", op_type="Add"),
  }
  edges = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),
  ]
  graph = LogicalGraph(name="AddNet", nodes=nodes, edges=edges)
  generator = IrToCstGenerator()
  module = generator.generate(graph)
  assert "self.add = nn.Add()" in module.code
  assert "add = self.add(in1)" in module.code
