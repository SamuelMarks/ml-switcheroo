"""
SASS Lifter (Frontend).

This module provides the logic to "lift" low-level SASS AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
to reconstruct the control flow and layer definitions.

Updates:
- Integrates `SassAnalyzer` to parse instructions between BEGIN/END markers.
- Populates `LogicalNode.metadata` with extracted parameters.
- **FIX**: Captures top-level instructions (unmapped/raw) into default logic blocks.
- **FIX**: Preserves register destinations as node IDs for faithful variables.
"""

import re
from typing import List, Optional, Pattern, Set

from ml_switcheroo.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.compiler.frontends.sass.nodes import (
  Comment,
  Instruction,
  SassNode,
  Register,
)
from ml_switcheroo.compiler.frontends.sass.analysis import SassAnalyzer


class SassLifter:
  """
  Reconstructs a LogicalGraph from a sequence of SASS AST nodes.
  """

  _RE_INPUT: Pattern = re.compile(r"Input\s+(\w+)\s+->")
  _RE_BEGIN: Pattern = re.compile(r"BEGIN\s+(\w+)\s+\((\w+)\)")
  _RE_END: Pattern = re.compile(r"END\s+(\w+)\s+\((\w+)\)")
  _RE_UNMAPPED: Pattern = re.compile(r"Unmapped Op:\s+([\w\.]+)\s+\((\w+)\)")
  _RE_RETURN: Pattern = re.compile(r"Return:")

  def lift(self, nodes: List[SassNode]) -> LogicalGraph:
    """
    Parses a list of SASS nodes to build a LogicalGraph.

    Captures instructions within BEGIN/END blocks to feed into the Analyzer.
    Captures orphan instructions into individual functional nodes (1:1 mapping).
    """
    # Default name matches test expectation for decompiled class
    graph = LogicalGraph(name="DecompiledModel")
    previous_node_id: Optional[str] = None
    seen_ids = set()

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

    instruction_counter = 0

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

        # Block End (Trigger Analysis)
        match_end = self._RE_END.search(text)
        if match_end:
          target_id = match_end.group(2)
          if target_id == current_block_id and current_block_kind:
            # Analyze collected instructions
            meta = SassAnalyzer.analyze_block(current_block_kind, current_instructions)
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
          # For unmapped, we assume default args (no instructions available)
          # Special Case: Flatten default start_dim=1 in PyTorch context
          meta = {}
          if "flatten" in api:
            meta["arg_1"] = 1

          commit_node(nid, api, meta)
          continue

        # Return
        if self._RE_RETURN.search(text):
          if "output" not in seen_ids:
            # No Logic, simple sink
            graph.nodes.append(LogicalNode(id="output", kind="Output"))
            if previous_node_id:
              graph.edges.append(LogicalEdge(source=previous_node_id, target="output"))
            seen_ids.add("output")
          continue

      # 2. Accumulate Instructions if inside a block
      if current_block_id is not None and isinstance(node, Instruction):
        current_instructions.append(node)

      # 3. Capture Orphan Instructions (Implicit 1:1 Ops)
      elif isinstance(node, Instruction):
        # Identify destination register to use as Node ID (Variable Name)
        # Heuristic: First operand of ALU ops is destination
        dest_name = f"inst_{instruction_counter}"
        is_alu = node.opcode.upper() in ["FADD", "FMUL", "IADD3", "FFMA", "MOV"]

        if is_alu and node.operands and isinstance(node.operands[0], Register):
          dest_name = node.operands[0].name

        instruction_counter += 1

        # Use 'asm' prefix to match test expectation "asm.FADD"
        kind = f"asm.{node.opcode}"

        # We construct metadata for arguments based on source operands
        meta = {}
        for i, op in enumerate(node.operands):
          # Skip first if dest?
          # Let's map all operands as arg_i for completeness
          meta[f"arg_{i}"] = str(op)

        commit_node(dest_name, kind, meta)

    return graph
