"""Test suite for the Sharding Extractor module."""

import typing

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass() -> None:
  """Docstring."""
  nodes = {
    "x": LogicalNode(id="x", op_type="Input"),
    "fc": LogicalNode(id="fc", op_type="Linear"),
    "func_sharding": LogicalNode(
      id="func_sharding",
      op_type="jax.lax.with_sharding_constraint",
      attributes={"arg_0": "x", "arg_1": "jax.sharding.PartitionSpec('data', None)"},
    ),
    "out": LogicalNode(id="out", op_type="Output"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("x", "fc"), LogicalEdge("fc", "func_sharding"), LogicalEdge("func_sharding", "out")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  node_ids: set[str] = set(extracted_graph.nodes.keys())
  assert "func_sharding" not in node_ids
  assert "fc" in node_ids
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extracted_graph.edges]
  assert ("fc", "out") in edges
  fc_node: LogicalNode = extracted_graph.nodes["fc"]
  assert fc_node.sharding is not None
  assert fc_node.sharding.axes == ("data", None)


def test_sharding_extraction_pass_complex_spec() -> None:
  """Docstring."""
  nodes = {
    "fc": LogicalNode(id="fc", op_type="Linear"),
    "func_sharding": LogicalNode(
      id="func_sharding",
      op_type="with_sharding_constraint",
      attributes={"arg_1": "PartitionSpec('data', ('model', 'tensor'))"},
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  fc_node: LogicalNode = extracted_graph.nodes["fc"]
  assert fc_node.sharding is not None
  assert fc_node.sharding.axes == ("data", ("model", "tensor"))


def test_sharding_extraction_pass_no_match() -> None:
  """Docstring."""
  graph = LogicalGraph(nodes={"fc": LogicalNode(id="fc", op_type="Linear")})
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  assert len(extracted_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_ast() -> None:
  """Docstring."""
  nodes = {
    "fc": LogicalNode(id="fc", op_type="Linear"),
    "func_sharding": LogicalNode(
      id="func_sharding", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', "}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  fc_node: LogicalNode = extracted_graph.nodes["fc"]
  assert fc_node.sharding is None


def test_sharding_extraction_pass_fallback_arg() -> None:
  """Docstring."""
  nodes = {
    "fc": LogicalNode(id="fc", op_type="Linear"),
    "func_sharding": LogicalNode(
      id="func_sharding", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', [1, 2])"}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  fc_node: LogicalNode = extracted_graph.nodes["fc"]
  assert fc_node.sharding is not None
  assert fc_node.sharding.axes == ("data", None)


def test_sharding_extraction_pass_no_partition_spec() -> None:
  """Docstring."""
  nodes = {
    "fc": LogicalNode(id="fc", op_type="Linear"),
    "func_sharding": LogicalNode(
      id="func_sharding", op_type="with_sharding_constraint", attributes={"arg_1": "SomeOtherConstraint()"}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph: LogicalGraph = pass_.apply(graph)
  assert "func_sharding" in set(extracted_graph.nodes.keys())


def test_sharding_extractor_no_source() -> None:
  """Verifies the behavior when source node is not found."""
  graph = LogicalGraph(
    nodes={
      "sharding1": LogicalNode(
        id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data')"}
      )
    },
    edges=[],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  # should not remove anything since source not found
  assert len(extracted.nodes) == 1


def test_sharding_extractor_invalid_ast() -> None:
  """Verifies the behavior when AST parsing fails."""
  nodes = {
    "source": LogicalNode(id="source", op_type="Linear"),
    "sharding1": LogicalNode(id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec("}),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 2


def test_sharding_extractor_no_partition_spec() -> None:
  """Verifies the behavior when PartitionSpec is not in code."""
  nodes = {
    "source": LogicalNode(id="source", op_type="Linear"),
    "sharding1": LogicalNode(
      id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "something_else()"}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 2


def test_sharding_extractor_tuple_arg() -> None:
  """Verifies the behavior when PartitionSpec has tuple."""
  nodes = {
    "source": LogicalNode(id="source", op_type="Linear"),
    "sharding1": LogicalNode(
      id="sharding1",
      op_type="with_sharding_constraint",
      attributes={"arg_1": "PartitionSpec('data', ('model', 'tensor'))"},
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 1
  assert extracted.nodes["source"].sharding is not None
  assert extracted.nodes["source"].sharding.axes == ("data", ("model", "tensor"))


def test_sharding_extractor_none_arg() -> None:
  """Verifies the behavior when PartitionSpec has None."""
  nodes = {
    "source": LogicalNode(id="source", op_type="Linear"),
    "sharding1": LogicalNode(
      id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec(None)"}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 1
  assert extracted.nodes["source"].sharding is not None
  assert extracted.nodes["source"].sharding.axes == (None,)


def test_sharding_extractor_unsupported_arg() -> None:
  """Verifies the behavior when PartitionSpec has unsupported arg type."""
  nodes = {
    "source": LogicalNode(id="source", op_type="Linear"),
    "sharding1": LogicalNode(
      id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec(unsupported_var)"}
    ),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 1
  assert extracted.nodes["source"].sharding is not None
  assert extracted.nodes["source"].sharding.axes == (None,)


def test_sharding_extractor_source_node_not_found() -> None:
  """Verifies the behavior when source node id is found but node object not in graph."""
  graph = LogicalGraph(
    nodes={
      "sharding1": LogicalNode(
        id="sharding1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data')"}
      )
    },
    edges=[LogicalEdge("missing_source", "sharding1")],
  )
  pass_ = ShardingExtractionPass()
  extracted: LogicalGraph = pass_.apply(graph)
  assert len(extracted.nodes) == 1


def test_sharding_extractor_duplicate_edge() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  nodes = {
    "A": LogicalNode("A", op_type="Op"),
    "Shard": LogicalNode("Shard", op_type="jax.lax.with_sharding_constraint", attributes={"arg_1": "PartitionSpec()"}),
    "B": LogicalNode("B", op_type="Op"),
  }
  edges = [
    LogicalEdge("A", "Shard"),
    LogicalEdge("Shard", "B"),
    LogicalEdge("A", "B"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  ShardingExtractionPass().apply(g)


def test_sharding_extractor_ast_not_call() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  # The string must contain PartitionSpec to pass the substring check
  res: typing.Any = ShardingExtractionPass()._parse_partition_spec("'PartitionSpec'")
  assert res is None


def test_sharding_extractor_duplicate_edge_not_in_new_edges() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  nodes = {
    "A": LogicalNode("A", op_type="Op"),
    "Shard": LogicalNode("Shard", op_type="jax.lax.with_sharding_constraint", attributes={"arg_1": "PartitionSpec()"}),
    "B": LogicalNode("B", op_type="Op"),
  }
  edges = [
    LogicalEdge("A", "Shard"),
    LogicalEdge("A", "B"),
    LogicalEdge("Shard", "B"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  ShardingExtractionPass().apply(g)


def test_sharding_extractor_duplicate_edge_not_in_new_edges_exact() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  nodes = {
    "A": LogicalNode("A", op_type="Op"),
    "Shard": LogicalNode("Shard", op_type="jax.lax.with_sharding_constraint", attributes={"arg_1": "PartitionSpec()"}),
    "B": LogicalNode("B", op_type="Op"),
  }
  edges = [
    LogicalEdge("A", "Shard"),
    LogicalEdge("A", "B"),
    LogicalEdge("Shard", "B"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  ShardingExtractionPass().apply(g)


def test_sharding_extractor_duplicate_edge_not_in_new_edges_exact_dataclass() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  nodes = {
    "A": LogicalNode("A", op_type="Op"),
    "Shard": LogicalNode("Shard", op_type="jax.lax.with_sharding_constraint", attributes={"arg_1": "PartitionSpec()"}),
    "B": LogicalNode("B", op_type="Op"),
  }
  edges = [
    LogicalEdge(source="A", target="Shard"),
    LogicalEdge(source="A", target="B"),
    LogicalEdge(source="Shard", target="B"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  ShardingExtractionPass().apply(g)


def test_sharding_extractor_duplicate_edge_2() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass

  nodes = {
    "source": LogicalNode("source", op_type="Linear"),
    "shard": LogicalNode("shard", op_type="with_sharding_constraint", attributes={"mesh": "a"}),
    "target": LogicalNode("target", op_type="relu"),
  }
  edges = [
    LogicalEdge("source", "shard"),
    LogicalEdge("shard", "target"),
    LogicalEdge("shard", "target"),
    LogicalEdge("source", "target"),
  ]
  graph = LogicalGraph(name="test", nodes=nodes, edges=edges)

  pass_ = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  edges_res: list[tuple[str, str]] = [(e.source, e.target) for e in new_graph.edges]
  assert edges_res.count(("source", "target")) == 1
