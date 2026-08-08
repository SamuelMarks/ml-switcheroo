"""RDNA Lifter (Frontend).

This module provides the logic to "lift" low-level RDNA AST nodes back into a
high-level `LogicalGraph`. It relies on semantic comment markers (e.g. `; BEGIN`)
emitted during compilation to reconstruct the control flow and layer definitions.
"""

from typing import Any

from typing import List, Optional, Set

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaInstruction,
  RdnaNode,
)
from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticCommentParser,
  SemanticInput,
  SemanticBegin,
  SemanticEnd,
  SemanticUnmapped,
  SemanticReturn,
)


class RdnaLifter:
  """Reconstructs a LogicalGraph from a sequence of RDNA AST nodes.

  This lifter processes sequential RDNA abstract syntax tree (AST) nodes,
  interprets semantic comment markers, and organizes raw instructions into a
  graph representation suitable for high-level compiler analysis.
  """

  def __init__(self) -> None:
    """Initialize the RdnaLifter."""
    self.comment_parser = SemanticCommentParser()

  def lift(self, nodes: List[RdnaNode]) -> LogicalGraph:
    """Parses a list of RDNA nodes to build a LogicalGraph.

    Args:
        nodes (List[RdnaNode]): A list of low-level RDNA AST nodes (instructions,
            comments, etc.) to process.

    Returns:
        LogicalGraph: The reconstructed high-level logical computation graph.
    """
    # Set default name to match expected Python Class Name in tests
    graph = LogicalGraph(name="DecompiledNet")
    previous_node_id: Optional[str] = None
    seen_ids: Set[str] = set()

    # State for block capture
    current_block_id: Optional[str] = None
    current_block_kind: Optional[str] = None
    current_instructions: List[RdnaInstruction] = []

    instruction_counter = 0

    def commit_node(node_id: str, kind: str, meta: Any = None) -> None:
      """Creates a logical node and appends it to the graph.

      Also creates a logical edge connecting the previously committed node to
      the newly created node to capture the control flow.

      Args:
          node_id (str): The unique identifier for the new logical node.
          kind (str): The operation kind of the new logical node.
          meta: Metadata dictionary for the logical node.
              Defaults to None.

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

    for node in nodes:
      if isinstance(node, RdnaComment):
        text = node.text.lstrip(";/ ").strip()
        marker = self.comment_parser.parse(text)

        if not marker:
          continue

        if isinstance(marker, SemanticInput):
          commit_node(marker.name, "Input", {"name": marker.name})
          continue

        if isinstance(marker, SemanticBegin):
          current_block_kind = marker.kind
          current_block_id = marker.id
          current_instructions = []
          continue

        if isinstance(marker, SemanticEnd):
          if marker.id == current_block_id and current_block_kind:
            meta = RdnaAnalyzer.analyze_block(current_block_kind, current_instructions)
            commit_node(current_block_id, current_block_kind, meta)
            current_block_id = None
            current_block_kind = None
            current_instructions = []
          continue

        if isinstance(marker, SemanticUnmapped):
          meta = {}
          if "flatten" in marker.api:
            meta["arg_1"] = 1
          commit_node(marker.id, marker.api, meta)
          continue

        if isinstance(marker, SemanticReturn):
          if "output" not in seen_ids:
            graph.nodes.append(LogicalNode(id="output", kind="Output"))
            if previous_node_id:
              graph.edges.append(LogicalEdge(source=previous_node_id, target="output"))
            seen_ids.add("output")
          continue

      if current_block_id is not None and isinstance(node, RdnaInstruction):
        current_instructions.append(node)

      elif isinstance(node, RdnaInstruction):
        op_id = f"inst_{instruction_counter}"
        instruction_counter += 1
        kind = f"rdna.{node.opcode}"
        commit_node(op_id, kind, {})

    return graph
