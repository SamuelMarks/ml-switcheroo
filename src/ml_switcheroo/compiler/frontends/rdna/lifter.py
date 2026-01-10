"""
RDNA Lifter (Frontend).

This module provides the logic to "lift" low-level RDNA AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
emitted during compilation to reconstruct the control flow and layer definitions.
"""

import re
from typing import List, Optional, Pattern, Set

from ml_switcheroo.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Comment,
  Instruction,
  RdnaNode,
)


class RdnaLifter:
  """
  Reconstructs a LogicalGraph from a sequence of RDNA AST nodes.
  """

  _RE_INPUT: Pattern = re.compile(r"Input\s+(\w+)\s+->")
  _RE_BEGIN: Pattern = re.compile(r"BEGIN\s+(\w+)\s+\((\w+)\)")
  _RE_END: Pattern = re.compile(r"END\s+(\w+)\s+\((\w+)\)")
  _RE_UNMAPPED: Pattern = re.compile(r"Unmapped Op:\s+([\w\.]+)\s+\((\w+)\)")
  _RE_RETURN: Pattern = re.compile(r"Return:")

  def lift(self, nodes: List[RdnaNode]) -> LogicalGraph:
    """
    Parses a list of RDNA nodes to build a LogicalGraph.
    """
    # Set default name to match expected Python Class Name in tests
    graph = LogicalGraph(name="DecompiledNet")
    previous_node_id: Optional[str] = None
    seen_ids: Set[str] = set()

    # State for block capture
    current_block_id: Optional[str] = None
    current_block_kind: Optional[str] = None
    current_instructions: List[Instruction] = []

    instruction_counter = 0

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

        # Block End
        match_end = self._RE_END.search(text)
        if match_end:
          target_id = match_end.group(2)
          if target_id == current_block_id and current_block_kind:
            meta = RdnaAnalyzer.analyze_block(current_block_kind, current_instructions)
            commit_node(current_block_id, current_block_kind, meta)
            current_block_id = None
            current_block_kind = None
            current_instructions = []
          continue

        # Unmapped
        match_unmapped = self._RE_UNMAPPED.search(text)
        if match_unmapped:
          api = match_unmapped.group(1)
          nid = match_unmapped.group(2)
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

      if current_block_id is not None and isinstance(node, Instruction):
        current_instructions.append(node)

      elif isinstance(node, Instruction):
        op_id = f"inst_{instruction_counter}"
        instruction_counter += 1
        kind = f"rdna.{node.opcode}"
        commit_node(op_id, kind, {})

    return graph
