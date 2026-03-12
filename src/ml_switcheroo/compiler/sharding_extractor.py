"""Sharding Extraction Pass (Reverse Translation).

This module implements a pass to extract `with_sharding_constraint` operations
from the `LogicalGraph` back into explicit `sharding` metadata on the source nodes.
"""

from typing import Dict, List, Optional
import ast
import re

from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec


class ShardingExtractionPass:
  """Extracts inline sharding constraints and updates node metadata."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to extract sharding constraints."""
    sharding_nodes = {n.id: n for n in graph.nodes if "with_sharding_constraint" in n.kind}

    if not sharding_nodes:
      return graph

    # Maps a sharding node's ID to its target source node
    removal_map = {}
    edges = graph.edges

    for sid, snode in sharding_nodes.items():
      # Find the source node that feeds into the sharding node
      source_id = None
      for e in edges:
        if e.target == sid:
          source_id = e.source
          break

      if not source_id:
        continue

      # Find the source node object
      source_node = None
      for n in graph.nodes:
        if n.id == source_id:
          source_node = n
          break

      if not source_node:
        continue

      # Parse PartitionSpec from metadata
      arg1 = snode.metadata.get("arg_1", "")

      # Minimal parsing of "jax.sharding.PartitionSpec('data', None)" or similar
      # It could be 'PartitionSpec(...)'
      spec = self._parse_partition_spec(arg1)
      if spec:
        source_node.sharding = spec
        removal_map[sid] = source_id

    if removal_map:
      new_nodes = [n for n in graph.nodes if n.id not in removal_map]
      graph.nodes = new_nodes

      new_edges = []
      for e in graph.edges:
        if e.target in removal_map:
          # Edge from source to sharding constraint node is removed
          pass
        elif e.source in removal_map:
          # Edge from sharding constraint to output is wired from original source
          new_source = removal_map[e.source]
          new_edge = LogicalEdge(source=new_source, target=e.target)
          if new_edge not in new_edges:
            new_edges.append(new_edge)
        else:
          new_edges.append(e)

      graph.edges = new_edges

    return graph

  def _parse_partition_spec(self, code: str) -> Optional[PartitionSpec]:
    """Extracts tuple from PartitionSpec string via AST."""
    if not ("PartitionSpec" in code or "NamedSharding" in code):
      return None

    try:
      # Use AST to parse the arguments safely
      tree = ast.parse(code, mode="eval")
      if isinstance(tree.body, ast.Call):
        axes = []
        for arg in tree.body.args:
          if isinstance(arg, ast.Constant):
            if arg.value is None:
              axes.append(None)
            else:
              axes.append(arg.value)
          elif isinstance(arg, ast.Tuple):
            tup = tuple(el.value for el in arg.elts if isinstance(el, ast.Constant))
            axes.append(tup)
          else:
            # Fallback
            axes.append(None)
        return PartitionSpec(axes=tuple(axes))
    except Exception:
      pass
    return None
