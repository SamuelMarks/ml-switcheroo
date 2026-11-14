"""Unit tests for `ShardingExtractionPass` (`ml_switcheroo.core.compiler.sharding_extractor`)."""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass_rewiring_deduplication_and_edge_cases() -> None:
  """Verify `ShardingExtractionPass` absorbs `with_sharding_constraint`, rewires edges, and deduplicates."""
  pass_ = ShardingExtractionPass()

  # Full chain `x -> fc -> func_sharding -> out` + pre-existing duplicate `fc -> out` + unrelated `u1 -> u2`
  graph = LogicalGraph(
    nodes={
      "x": LogicalNode("x", "Input"),
      "fc": LogicalNode("fc", "Linear"),
      "func_sharding": LogicalNode(
        "func_sharding",
        "jax.lax.with_sharding_constraint",
        attributes={"arg_0": "x", "arg_1": "jax.sharding.PartitionSpec('data', None)"},
      ),
      "out": LogicalNode("out", "Output"),
      "u1": LogicalNode("u1", "Other"),
      "u2": LogicalNode("u2", "Other"),
    },
    edges=[
      LogicalEdge("x", "fc"),
      LogicalEdge("fc", "func_sharding"),
      LogicalEdge("fc", "out"),
      LogicalEdge("func_sharding", "out"),
      LogicalEdge("func_sharding", "out"),
      LogicalEdge("u1", "u2"),
    ],
  )
  extracted = pass_.apply(graph)
  assert "func_sharding" not in extracted.nodes
  assert extracted.nodes["fc"].sharding == PartitionSpec(axes=("data", None))
  edge_pairs = [(e.source, e.target) for e in extracted.edges]
  assert edge_pairs.count(("fc", "out")) == 1
  assert ("u1", "u2") in edge_pairs

  # Reverse edge ordering (`fc -> func_sharding` then `func_sharding -> out` BEFORE `fc -> out`)
  rev = pass_.apply(
    LogicalGraph(
      nodes={
        "fc": LogicalNode("fc", "Linear"),
        "s": LogicalNode("s", "with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data')"}),
        "out": LogicalNode("out", "Output"),
      },
      edges=[LogicalEdge("fc", "s"), LogicalEdge("s", "out"), LogicalEdge("fc", "out")],
    )
  )
  assert ("fc", "out") in [(e.source, e.target) for e in rev.edges]

  # Non-matching graph, missing incoming edge, missing source node in `nodes`, and unparseable spec
  assert len(pass_.apply(LogicalGraph(nodes={"fc": LogicalNode("fc", "Linear")})).nodes) == 1
  assert (
    len(
      pass_.apply(
        LogicalGraph(
          nodes={"s": LogicalNode("s", "with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data')"})}
        )
      ).nodes
    )
    == 1
  )
  assert (
    len(
      pass_.apply(
        LogicalGraph(
          nodes={"s": LogicalNode("s", "with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data')"})},
          edges=[LogicalEdge("missing_src", "s")],
        )
      ).nodes
    )
    == 1
  )
  assert (
    len(
      pass_.apply(
        LogicalGraph(
          nodes={
            "fc": LogicalNode("fc", "Linear"),
            "s": LogicalNode("s", "with_sharding_constraint", attributes={"arg_1": "PartitionSpec(!"}),
          },
          edges=[LogicalEdge("fc", "s")],
        )
      ).nodes
    )
    == 2
  )


def test_parse_partition_spec_variants() -> None:
  """Verify `_parse_partition_spec` handles nested tuples, `NamedSharding`, `None`, non-calls, and invalid syntax."""
  pass_ = ShardingExtractionPass()
  assert pass_._parse_partition_spec("SomeOtherConstraint()") is None
  assert pass_._parse_partition_spec("'PartitionSpec'") is None
  assert pass_._parse_partition_spec("PartitionSpec(") is None
  assert pass_._parse_partition_spec("PartitionSpec('tensor')") == PartitionSpec(axes=("tensor",))
  assert pass_._parse_partition_spec("NamedSharding('tensor')") == PartitionSpec(axes=("tensor",))
  assert pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)") == PartitionSpec(
    axes=(("data", "tensor"), None)
  )
  assert pass_._parse_partition_spec("PartitionSpec('data', [1, 2])") == PartitionSpec(axes=("data", None))
  assert pass_._parse_partition_spec("PartitionSpec(unsupported_var)") == PartitionSpec(axes=(None,))
