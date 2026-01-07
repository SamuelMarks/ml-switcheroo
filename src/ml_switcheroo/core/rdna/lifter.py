"""
RDNA Lifter Module.

This module provides the logic to "lift" low-level RDNA AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
emitted during compilation to reconstruct the control flow and layer definitions.

It integrates `RdnaAnalyzer` to reverse-engineer layer properties (kernel sizes)
from the assembly instructions found within the marked blocks.
"""

import re
from typing import List, Optional, Pattern, Set

from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.rdna.nodes import Comment, Instruction, RdnaNode


class RdnaLifter:
  """
  Reconstructs a LogicalGraph from a sequence of RDNA AST nodes.
  """

  # Regex definitions for markers
  _RE_INPUT: Pattern = re.compile(r"Input\s+(\w+)\s+->")
  _RE_BEGIN: Pattern = re.compile(r"BEGIN\s+(\w+)\s+\((\w+)\)")
  _RE_END: Pattern = re.compile(r"END\s+(\w+)\s+\((\w+)\)")
  _RE_UNMAPPED: Pattern = re.compile(r"Unmapped Op:\s+([\w\.]+)\s+\((\w+)\)")
  _RE_RETURN: Pattern = re.compile(r"Return:")

  def lift(self, nodes: List[RdnaNode]) -> LogicalGraph:
    """
    Parses a list of RDNA nodes to build a LogicalGraph.

    Captures instructions within BEGIN/END blocks to feed into the Analyzer.

    Args:
        nodes (List[RdnaNode]): The parsed AST.

    Returns:
        LogicalGraph: The reconstructed graph model.
    """
    graph = LogicalGraph()
    previous_node_id: Optional[str] = None
    seen_ids: Set[str] = set()

    # State for block capture
    current_block_id: Optional[str] = None
    current_block_kind: Optional[str] = None
    current_instructions: List[Instruction] = []

    def commit_node(node_id: str, kind: str, meta=None) -> None:
      nonlocal previous_node_id
      if node_id in seen_ids:
        return

      node = LogicalNode(id=node_id, kind=kind, metadata=meta or {})
      graph.nodes.append(node)
      seen_ids.add(node_id)

      if previous_node_id:
        graph.edges.append(LogicalEdge(source=previous_node_id, target=node_id))
      previous_node_id = node_id

    for node in nodes:
      # 1. Processing Comments (Markers)
      if isinstance(node, Comment):
        text = node.text.strip()

        # Input
        match_input = self._RE_INPUT.search(text)
        if match_input:
          var_name = match_input.group(1)
          commit_node(var_name, "Input", {"name": var_name})
          continue

        # Block Start
        match_begin = self._RE_BEGIN.search(text)
        if match_begin:
          current_block_kind = match_begin.group(1)
          current_block_id = match_begin.group(2)
          current_instructions = []
          continue

        # Block End (Trigger Analysis)
        match_end = self._RE_END.search(text)
        if match_end:
          target_id = match_end.group(2)
          # Close the currently open block if IDs match
          if target_id == current_block_id and current_block_kind:
            # Analyze collected instructions
            meta = RdnaAnalyzer.analyze_block(current_block_kind, current_instructions)
            commit_node(current_block_id, current_block_kind, meta)

            # Reset
            current_block_id = None
            current_block_kind = None
            current_instructions = []
          continue

        # Unmapped Ops
        match_unmapped = self._RE_UNMAPPED.search(text)
        if match_unmapped:
          api = match_unmapped.group(1)
          nid = match_unmapped.group(2)
          # Default heuristic for flattening if found
          meta = {}
          if "flatten" in api:
            meta["arg_1"] = 1

          commit_node(nid, api, meta)
          continue

        # Return
        if self._RE_RETURN.search(text):
          if "output" not in seen_ids:
            graph.nodes.append(LogicalNode(id="output", kind="Output"))
            if previous_node_id:
              graph.edges.append(LogicalEdge(source=previous_node_id, target="output"))
            seen_ids.add("output")
          continue

      # 2. Accumulate Instructions if inside a block
      if current_block_id is not None and isinstance(node, Instruction):
        current_instructions.append(node)

    return graph
