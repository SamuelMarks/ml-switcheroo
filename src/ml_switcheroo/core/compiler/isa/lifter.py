"""Unified ISA Lifter (Frontend).

Provides a generic base class `IsaLifter` that reconstructs a high-level `LogicalGraph`
from a sequence of low-level hardware ISA AST nodes (`NvidiaSassNode`, `RdnaNode`, etc.)
using `SemanticCommentParser` markers (`BEGIN`, `END`, ` INPUT`, `RETURN`, `Unmapped Op`)
and `IsaAnalyzer` parameter recovery.
"""

import typing

from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticBegin,
  SemanticCommentParser,
  SemanticEnd,
  SemanticInput,
  SemanticReturn,
  SemanticUnmapped,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer
from ml_switcheroo.core.compiler.isa.cst import IsaComment, IsaInstruction, IsaNode

NodeT = typing.TypeVar("NodeT", bound=IsaNode)
InstructionT = typing.TypeVar("InstructionT", bound=IsaInstruction)
CommentT = typing.TypeVar("CommentT", bound=IsaComment)


class IsaLifter(typing.Generic[NodeT, InstructionT, CommentT]):
  """Generic base lifter that reconstructs a `LogicalGraph` from ISA AST nodes."""

  DEFAULT_GRAPH_NAME: str = "DecompiledModel"
  COMMENT_TYPE: type[IsaComment] = IsaComment
  INSTRUCTION_TYPE: type[IsaInstruction] = IsaInstruction
  ANALYZER_CLS: type[IsaAnalyzer] = IsaAnalyzer
  ORPHAN_PREFIX: str = "asm"

  def __init__(self) -> None:
    """Initialize the ISA lifter and its semantic comment parser."""
    self.comment_parser = SemanticCommentParser()

  def _resolve_orphan_instruction(
    self,
    node: InstructionT,
    instruction_counter: int,
  ) -> tuple[str, str, dict[str, str | int | float]]:
    """Resolve an unenclosed (orphan) instruction into a `(node_id, kind, metadata)` tuple.

    Args:
        node: The raw ISA instruction node.
        instruction_counter: Monotonic counter for unnamed instructions.

    Returns:
        Tuple of `(node_id, op_kind, metadata_dict)`.
    """
    opcode = getattr(node, "opcode", "unknown")
    return (f"inst_{instruction_counter}", f"{self.ORPHAN_PREFIX}.{opcode}", {})

  def _is_instruction(self, node: IsaNode) -> typing.TypeGuard[InstructionT]:
    """Check whether an ISA AST node is an instruction of this lifter's instruction type.

    Args:
        node: Candidate ISA AST node.

    Returns:
        True if ``node`` is an instance of ``self.INSTRUCTION_TYPE``.
    """
    return isinstance(node, self.INSTRUCTION_TYPE)

  def lift(self, nodes: typing.Sequence[IsaNode]) -> LogicalGraph:
    """Parse a sequence of ISA AST nodes to build a `LogicalGraph`.

    Args:
        nodes: Sequence of low-level ISA AST nodes (comments, instructions, labels).

    Returns:
        Reconstructed high-level `LogicalGraph`.
    """
    graph = LogicalGraph(name=self.DEFAULT_GRAPH_NAME)
    previous_node_id: str | None = None
    seen_ids: set[str] = set()

    current_block_id: str | None = None
    current_block_kind: str | None = None
    current_instructions: list[InstructionT] = []
    instruction_counter = 0

    def commit_node(
      node_id: str,
      kind: str,
      meta: typing.Mapping[str, str | int | float] | None = None,
    ) -> None:
      """Commit a LogicalNode and its control/dataflow edge to the graph.

      Args:
          node_id: Unique identifier for the logical node.
          kind: Operation type represented by the node.
          meta: Optional metadata dictionary of extracted attributes.
      """
      nonlocal previous_node_id
      if node_id in seen_ids:
        return

      node_obj = LogicalNode(id=node_id, op_type=kind, attributes=dict(meta) if meta else {})
      graph.nodes[node_id] = node_obj
      seen_ids.add(node_id)

      if previous_node_id:
        graph.add_edge(previous_node_id, node_id)
      previous_node_id = node_id

    for node in nodes:
      if isinstance(node, self.COMMENT_TYPE):
        text = getattr(node, "text", "").lstrip(";/ ").strip()
        marker = self.comment_parser.parse(text)
        if not marker:
          continue

        if isinstance(marker, SemanticInput):
          commit_node(marker.name, "Input", {"name": marker.name})
        elif isinstance(marker, SemanticBegin):
          current_block_kind = marker.kind
          current_block_id = marker.id
          current_instructions = []
        elif isinstance(marker, SemanticEnd):
          if marker.id == current_block_id and current_block_kind:
            meta = self.ANALYZER_CLS.analyze_block(current_block_kind, current_instructions)
            commit_node(current_block_id, current_block_kind, meta)
            current_block_id = None
            current_block_kind = None
            current_instructions = []
        elif isinstance(marker, SemanticUnmapped):
          unmapped_meta: dict[str, str | int | float] = {"arg_1": 1} if "flatten" in marker.api else {}
          commit_node(marker.id, marker.api, unmapped_meta)
        elif isinstance(marker, SemanticReturn) and "output" not in seen_ids:
          graph.nodes["output"] = LogicalNode(id="output", op_type="Output")
          if previous_node_id:
            graph.add_edge(previous_node_id, "output")
          seen_ids.add("output")

      elif current_block_id is not None and self._is_instruction(node):
        current_instructions.append(node)
      elif self._is_instruction(node):
        dest_name, kind, meta = self._resolve_orphan_instruction(node, instruction_counter)
        instruction_counter += 1
        commit_node(dest_name, kind, meta)

    return graph
