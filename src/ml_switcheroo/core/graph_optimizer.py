"""Graph-Level Rewriter for Pattern Fusion.

This module implements the `GraphOptimizer`, which performs fusion on
the `LogicalGraph` intermediate representation. It scans the graph for
topological sequences that match predefined patterns (e.g., Conv -> BN -> ReLU)
and replaces them with fused macro nodes.

Algorithm:
    1.  Build an adjacency map for efficient traversal.
    2.  Iterate through all nodes in the graph in topological order.
    3.  For each node, check if it starts a sequence matching any known PatternDef.
    4.  If a match is found (A -> B -> C):
        a. Create a new Fused Node (Kind=Replacement).
        b. Merge metadata from A, B, C into Fused Node.
        c. Rewire input edges: Source(A) -> Fused.
        d. Rewire output edges: Fused -> Target(C).
        e. Mark A, B, C for removal.
        f. Continue scan after C.
    5.  Reconstruct graph with fused nodes and updated edges.
"""

import copy
from collections import defaultdict

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, topological_sort
from ml_switcheroo.core.dsl import PatternDef


class GraphOptimizer:
  """Optimize a LogicalGraph by fusing subgraphs based on defined patterns."""

  def __init__(self, patterns: list[PatternDef]):
    """Initialize with a list of fusion patterns.

    Args:
        patterns: List of `PatternDef` objects defining the sequences to fuse.

    """
    self.patterns = patterns

  def optimize(self, graph: LogicalGraph) -> LogicalGraph:
    """Run the fusion pass on the graph.

    Args:
        graph: The input `LogicalGraph`.

    Returns:
        A new `LogicalGraph` with fusion applied.

    """
    if not self.patterns:
      return graph

    # Work on a copy to avoid mutating source in place if retrying
    current_graph = copy.deepcopy(graph)

    return self._apply_fusion_pass(current_graph)

  @staticmethod
  def _reroute_fused_edges(
    edges: list[LogicalEdge],
    node_to_fused_id: dict[str, str],
    fusion_map: dict[str, dict[str, str]],
  ) -> list[LogicalEdge]:
    """Rewrite ``edges`` around newly fused pattern head/tail endpoints.

    Args:
        edges: Original graph edges prior to fusion.
        node_to_fused_id: Mapping from absorbed node IDs to their fused node ID.
        fusion_map: Mapping from fused node ID to ``{'head': id, 'tail': id}``.

    Returns:
        Rewritten list of external ``LogicalEdge`` connections.
    """
    final_edges: list[LogicalEdge] = []
    for e in edges:
      src_fused = node_to_fused_id.get(e.source)
      tgt_fused = node_to_fused_id.get(e.target)

      final_src = e.source if not src_fused else (src_fused if e.source == fusion_map[src_fused]["tail"] else None)
      final_tgt = e.target if not tgt_fused else (tgt_fused if e.target == fusion_map[tgt_fused]["head"] else None)

      if final_src and final_tgt and final_src != final_tgt:
        final_edges.append(LogicalEdge(source=final_src, target=final_tgt))
    return final_edges

  def _apply_fusion_pass(self, graph: LogicalGraph) -> LogicalGraph:
    """Execute a single pass of greedy pattern matching.

    Args:
        graph: The input graph.

    Returns:
        The optimized graph.

    """
    node_map = {n.id: n for n in graph.nodes.values()}
    out_edges: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
      out_edges[e.source].append(e.target)

    sorted_nodes = topological_sort(graph)
    node_to_fused_id: dict[str, str] = {}
    fusion_map: dict[str, dict[str, str]] = {}
    processed_ids: set[str] = set()
    new_nodes: list[LogicalNode] = []

    for node in sorted_nodes:
      if node.id in processed_ids:
        continue

      matched_pattern = None
      matched_ids: list[str] = []
      for pattern in self.patterns:
        ids = self._match_sequence(node, pattern.sequence, node_map, out_edges, processed_ids)
        if ids:
          matched_pattern = pattern
          matched_ids = ids
          break

      if matched_pattern and matched_ids:
        match_root, match_tail = matched_ids[0], matched_ids[-1]
        fused_id = f"fused_{match_root}"
        merged_meta: dict[str, AttributeValue] = {}
        for mid in matched_ids:
          merged_meta.update(node_map[mid].attributes)
        new_nodes.append(LogicalNode(id=fused_id, op_type=matched_pattern.replace_with, attributes=merged_meta))
        fusion_map[fused_id] = {"head": match_root, "tail": match_tail}
        for mid in matched_ids:
          processed_ids.add(mid)
          node_to_fused_id[mid] = fused_id
      else:
        new_nodes.append(node)

    final_edges = self._reroute_fused_edges(graph.edges, node_to_fused_id, fusion_map)
    for n in new_nodes:
      n.inputs = []

    return LogicalGraph(nodes={n.id: n for n in new_nodes}, edges=final_edges)

  def _match_sequence(
    self,
    start_node: LogicalNode,
    sequence: list[str],
    node_map: dict[str, LogicalNode],
    out_edges: dict[str, list[str]],
    processed_ids: set[str],
  ) -> list[str] | None:
    """Check if a sequence of Op Kinds exists starting from `start_node`.

    Enforces linear chain constraint (A->B->C).

    Args:
        start_node: The node to start matching from.
        sequence: List of operation kinds (strings).
        node_map: ID to Node lookup.
        out_edges: Adjacency map.
        processed_ids: Set of already consumed nodes (prevent re-consumption).

    Returns:
        List of node IDs forming access sequence, or None.

    """
    if not sequence:
      return None

    # Check first node type
    if start_node.op_type != sequence[0]:
      return None

    matched_ids = [start_node.id]
    current_id = start_node.id

    # Iterate remaining sequence
    for kind in sequence[1:]:
      # Get neighbors
      targets = out_edges[current_id]

      # Find a neighbor that matches the Kind
      candidate_id = None
      for tgt in targets:
        if tgt in processed_ids:
          continue
        tgt_node = node_map.get(tgt)
        if tgt_node and tgt_node.op_type == kind:
          candidate_id = tgt
          break

      if candidate_id:
        matched_ids.append(candidate_id)
        current_id = candidate_id
      else:
        return None

    return matched_ids
