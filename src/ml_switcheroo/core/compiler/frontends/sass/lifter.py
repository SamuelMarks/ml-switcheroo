"""SASS Lifter (Frontend).

This module provides the logic to "lift" low-level SASS AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
to reconstruct the control flow and layer definitions.

Updates:
- Integrates `SassAnalyzer` to parse instructions between BEGIN/END markers.
- Populates `LogicalNode.metadata` with extracted parameters.
- **FIX**: Captures top-level instructions (unmapped/raw) into default logic blocks.
- **FIX**: Preserves register destinations as node IDs for faithful variables.
"""

from typing import Any

from typing import List, Optional

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassInstruction,
  SassNode,
  SassRegister,
)
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticCommentParser,
  SemanticInput,
  SemanticBegin,
  SemanticEnd,
  SemanticUnmapped,
  SemanticReturn,
)


class SassLifter:
  """Reconstructs a LogicalGraph from a sequence of SASS AST nodes."""

  def __init__(self) -> None:
    """Initialize SassLifter.

    Initializes the class by instantiating the internal SemanticCommentParser helper.
    """
    self.comment_parser = SemanticCommentParser()

  def lift(self, nodes: List[SassNode]) -> LogicalGraph:
    """Parses a list of SASS nodes to build a LogicalGraph.

    Captures instructions within BEGIN/END blocks to feed into the Analyzer.
    Captures orphan instructions into individual functional nodes (1:1 mapping).

    Args:
        nodes: A list of SASS AST nodes (such as comments or instructions) to parse.

    Returns:
        A LogicalGraph representing the reconstructed high-level architecture of the SASS program.
    """
    # Default name matches test expectation for decompiled class
    graph = LogicalGraph(name="DecompiledModel")
    previous_node_id: Optional[str] = None
    seen_ids = set()

    # State for block capture
    current_block_id: Optional[str] = None
    current_block_kind: Optional[str] = None
    current_instructions: List[SassInstruction] = []

    def commit_node(node_id: str, kind: str, meta: Any = None) -> None:
      """Helper function to commit a LogicalNode and its transition edge to the graph.

      Checks if the node has already been seen to prevent duplication, instantiates
      a LogicalNode with the provided id, kind, and metadata, appends it to the graph,
      and establishes a LogicalEdge from the previously committed node to the current one.

      Args:
          node_id: The unique identifier for the LogicalNode.
          kind: The type or operation represented by the node.
          meta: Optional metadata dictionary containing attributes and parameters.
      """
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
      if isinstance(node, SassComment):
        text = node.text.lstrip(";/ ").strip()
        marker = self.comment_parser.parse(text)

        if not marker:
          continue

        if isinstance(marker, SemanticInput):
          commit_node(marker.name, "Input", {"name": marker.name})
          continue

        elif isinstance(marker, SemanticBegin):
          current_block_kind = marker.kind
          current_block_id = marker.id
          current_instructions = []
          continue

        elif isinstance(marker, SemanticEnd):
          if marker.id == current_block_id:
            # Analyze collected instructions
            assert current_block_kind is not None
            assert current_block_id is not None
            meta = SassAnalyzer.analyze_block(current_block_kind, current_instructions)
            commit_node(current_block_id, current_block_kind, meta)

            # Reset
            current_block_id = None
            current_block_kind = None
            current_instructions = []

        elif isinstance(marker, SemanticUnmapped):
          # For unmapped, we assume default args (no instructions available)
          # Special Case: Flatten default start_dim=1 in PyTorch context
          meta = {}
          if "flatten" in marker.api:
            meta["arg_1"] = 1

          commit_node(marker.id, marker.api, meta)
          continue

        elif isinstance(marker, SemanticReturn):
          if "output" not in seen_ids:
            # No Logic, simple sink
            graph.nodes.append(LogicalNode(id="output", kind="Output"))
            if "previous_node_id" in locals() and previous_node_id:
              graph.edges.append(LogicalEdge(source=previous_node_id, target="output"))
            seen_ids.add("output")

      # 2. Accumulate Instructions if inside a block
      if current_block_id is not None and isinstance(node, SassInstruction):
        current_instructions.append(node)

      # 3. Capture Orphan Instructions (Implicit 1:1 Ops)
      elif isinstance(node, SassInstruction):
        # Identify destination register to use as Node ID (Variable Name)
        # Heuristic: First operand of ALU ops is destination
        dest_name = f"inst_{instruction_counter}"
        is_alu = node.opcode.upper() in ["FADD", "FMUL", "IADD3", "FFMA", "MOV"]

        if is_alu and node.operands and isinstance(node.operands[0], SassRegister):
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
