"""
Graph-Level Rewriter for Pattern Fusion.

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

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import copy

from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.dsl import PatternDef


class GraphOptimizer:
  """
  Optimizes a LogicalGraph by fusing subgraphs based on defined patterns.
  """

  def __init__(self, patterns: List[PatternDef]):
    """
    Initialize with a list of fusion patterns.

    Args:
        patterns: List of `PatternDef` objects defining the sequences to fuse.
    """
    self.patterns = patterns

  def optimize(self, graph: LogicalGraph) -> LogicalGraph:
    """
    Runs the fusion pass on the graph.

    Args:
        graph: The input `LogicalGraph`.

    Returns:
        A new `LogicalGraph` with fusion applied.
    """
    if not self.patterns:
      return graph

    # Work on a copy to avoid mutating source in place if retrying
    current_graph = copy.deepcopy(graph)

    fused = self._apply_fusion_pass(current_graph)
    return fused

  def _apply_fusion_pass(self, graph: LogicalGraph) -> LogicalGraph:
    """
    Executes a single pass of greedy pattern matching.
    """
    # 1. Build Adjacency and Lookup Maps
    # node_id -> Node
    node_map = {n.id: n for n in graph.nodes}
    # source_id -> [target_id]
    out_edges = defaultdict(list)
    # target_id -> [source_id]
    in_edges = defaultdict(list)

    for e in graph.edges:
      out_edges[e.source].append(e.target)
      in_edges[e.target].append(e.source)

    from ml_switcheroo.core.graph import topological_sort

    sorted_nodes = topological_sort(graph)

    # Track which original node maps to which fused node (or itself if not fused)
    # However, since we process greedily, we can just track fusion events
    node_to_fused_id: Dict[str, str] = {}

    # Details of fusion: fused_id -> {head: id, tail: id}
    fusion_map: Dict[str, Dict[str, str]] = {}

    processed_ids: Set[str] = set()
    new_nodes: List[LogicalNode] = []

    # Iterate via sorted list
    for node in sorted_nodes:
      if node.id in processed_ids:
        continue

      # Try to match patterns starting at 'node'
      matched_pattern = None
      matched_ids = []

      for pattern in self.patterns:
        ids = self._match_sequence(node, pattern.sequence, node_map, out_edges, processed_ids)
        if ids:
          matched_pattern = pattern
          matched_ids = ids
          break  # Greedy

      if matched_pattern and matched_ids:
        match_root = matched_ids[0]
        match_tail = matched_ids[-1]
        fused_id = f"fused_{match_root}"

        # Merge Metadata
        merged_meta = {}
        for mid in matched_ids:
          mnode = node_map[mid]
          merged_meta.update(mnode.metadata)

        fused_node = LogicalNode(id=fused_id, kind=matched_pattern.replace_with, metadata=merged_meta)
        new_nodes.append(fused_node)

        fusion_map[fused_id] = {"head": match_root, "tail": match_tail}

        # Mark involved nodes
        for mid in matched_ids:
          processed_ids.add(mid)
          node_to_fused_id[mid] = fused_id

      else:
        # No match, keep node as is
        new_nodes.append(node)

    # Rebuild Edges
    # We iterate over original edges and effectively re-route them
    final_edges: List[LogicalEdge] = []

    for e in graph.edges:
      src = e.source
      tgt = e.target

      src_fused = node_to_fused_id.get(src)
      tgt_fused = node_to_fused_id.get(tgt)

      # Logic Table:
      # 1. src Not Fused, tgt Not Fused -> Keep edge intact.
      # 2. src Fused (F1), tgt Not Fused ->
      #    If src == F1.tail -> Add edge F1 -> tgt.
      #    Else (internal logic) -> Drop.
      # 3. src Not Fused, tgt Fused (F2) ->
      #    If tgt == F2.head -> Add edge src -> F2.
      #    Else (internal logic) -> Drop.
      # 4. src Fused (F1), tgt Fused (F2) ->
      #    If F1 == F2 -> Drop (Internal).
      #    If F1 != F2, src == F1.tail, tgt == F2.head -> Add edge F1 -> F2.

      final_src = None
      final_tgt = None
      should_add = False

      # Resolve Source
      if not src_fused:
        final_src = src
      else:
        info = fusion_map[src_fused]
        if src == info["tail"]:
          final_src = src_fused
        # Else: src is internal non-tail, this edge is invalid for graph flow unless
        # we support branching inside patterns (not supported yet).

      # Resolve Target
      if not tgt_fused:
        final_tgt = tgt
      else:
        info = fusion_map[tgt_fused]
        if tgt == info["head"]:
          final_tgt = tgt_fused
        # Else: target is internal non-head.

      if final_src and final_tgt and final_src != final_tgt:
        final_edges.append(LogicalEdge(source=final_src, target=final_tgt))

    return LogicalGraph(nodes=new_nodes, edges=final_edges)

  def _match_sequence(
    self,
    start_node: LogicalNode,
    sequence: List[str],
    node_map: Dict[str, LogicalNode],
    out_edges: Dict[str, List[str]],
    processed_ids: Set[str],
  ) -> Optional[List[str]]:
    """
    Checks if a sequence of Op Kinds exists starting from `start_node`.
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
    if start_node.kind != sequence[0]:
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
        if tgt_node and tgt_node.kind == kind:
          candidate_id = tgt
          break

      if candidate_id:
        matched_ids.append(candidate_id)
        current_id = candidate_id
      else:
        return None

    return matched_ids
