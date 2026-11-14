"""LaTeX visualizer backend for LogicalGraphs.

This module provides the LatexBackend, which compiles a LogicalGraph into
standalone LaTeX documents using the Machine Intelligence Definition Language (MIDL)
package format, suitable for high-quality PDF rendering of neural architectures.
"""

import typing

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.python_strategies.init import _INTERNAL_TRIVIA_KEYS
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.latex.nodes import (
  ComputeNode,
  InputNode,
  LatexNode,
  MemoryNode,
  ModelContainer,
  ReturnNode,
  StateOpNode,
)

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


@typing.runtime_checkable
class _PendingEdgesProtocol(typing.Protocol):
  """Protocol for LogicalGraph instances exposing `_pending_edges`.

  Attributes:
      _pending_edges: Deferred edges accumulated prior to node materialization.
  """

  _pending_edges: list[LogicalEdge]


class LatexBackend(CompilerBackend):
  """Transform Logical Graphs into MIDL LaTeX format.

  Generates structured Machine Intelligence Definition Language (MIDL) LaTeX documents
  from a logical graph.
  """

  def __init__(self, semantics: "SemanticsManager | None" = None) -> None:
    """Initialize the LaTeX backend with optional semantic knowledge.

    Args:
        semantics: Optional semantics manager for operator resolution.
    """
    super().__init__(semantics=semantics)

  def compile(self, graph: LogicalGraph) -> str:
    """Compile a LogicalGraph to a complete standalone LaTeX document.

    Args:
        graph: The logical graph to compile into MIDL LaTeX.

    Returns:
        str: A string containing the full standalone LaTeX document.
    """
    name = graph.name or "GeneratedNet"
    container = self._transcode_graph(graph, name)
    return self._wrap_document(container.to_latex())

  def _wrap_document(self, content: str) -> str:
    """Wrap raw LaTeX content into a standalone document structure.

    Args:
        content: The raw LaTeX content body representing the model.

    Returns:
        str: The complete standalone LaTeX document string including package imports.
    """
    nl = chr(10)
    comment_block = nl.join(
      [
        r"% ------------------------------------------------------------------",
        r"% [Requirement] midl.sty",
        r"% This document uses the Machine Intelligence Definition Language.",
        r"% Ensure 'midl.sty' is in your LaTeX path.",
        r"% ------------------------------------------------------------------",
        "",
      ]
    )
    header = nl.join(
      [
        r"\documentclass[tikz, border=10pt, landscape]{standalone}",
        r"\usepackage{midl}",
        r"\begin{document}",
        "",
      ]
    )
    footer = nl + r"\end{document}"
    return comment_block + header + content + footer

  @staticmethod
  def _build_memory_nodes(state_registry: dict[str, LogicalNode]) -> list[LatexNode]:
    """Create ``MemoryNode`` declarations for stateful non-boundary ``LogicalNode`` entries.

    Args:
        state_registry: Mapping of node IDs to ``LogicalNode`` instances.

    Returns:
        Ordered list of ``MemoryNode`` AST elements.
    """
    memories: list[LatexNode] = []
    for node_id, node in sorted(state_registry.items()):
      op_type = str(node.op_type)
      if op_type in ("Input", "Output") or node_id.startswith("func_"):
        continue
      config = {str(k): str(v) for k, v in node.attributes.items() if k not in _INTERNAL_TRIVIA_KEYS}
      memories.append(MemoryNode(node_id=node_id, op_type=op_type, config=config))
    return memories

  @staticmethod
  def _build_compute_node(
    step_id: str,
    op_type: str,
    args: list[str],
    node_data: LogicalNode | None,
  ) -> ComputeNode:
    """Construct a functional ``ComputeNode`` with formatted positional and keyword metadata.

    Args:
        step_id: Generated step identifier string.
        op_type: Raw logical operation type string.
        args: Incoming edge argument reference strings.
        node_data: Optional ``LogicalNode`` holding operation attributes.

    Returns:
        Constructed ``ComputeNode`` instance.
    """
    meta_args: list[str] = []
    if node_data:
      for k, v in node_data.attributes.items():
        if k in _INTERNAL_TRIVIA_KEYS:
          continue
        val_str = str(v)
        meta_args.append(val_str if k.startswith("arg") else f"{k}={val_str}")

    clean_type = op_type
    if clean_type.startswith("func_"):
      clean_type = clean_type[5:]
    elif "." in clean_type:
      clean_type = clean_type.split(".")[-1]
    return ComputeNode(step_id, clean_type.capitalize(), args + meta_args, "[_]")

  @staticmethod
  def _build_return_node(graph: LogicalGraph, id_map: dict[str, str]) -> ReturnNode:
    """Construct the terminal ``ReturnNode`` referencing the graph output producer.

    Args:
        graph: Source ``LogicalGraph``.
        id_map: Mapping of logical node IDs to emitted LaTeX step identifiers.

    Returns:
        Constructed ``ReturnNode`` instance.
    """
    sources_to_output = [e.source for e in graph.edges if e.target in ("output", "Output")]
    if sources_to_output:
      final_src = sources_to_output[0]
      return ReturnNode(target_id=id_map.get(final_src, f"op_{final_src}"))
    return ReturnNode(target_id="last_step")

  def _transcode_graph(self, graph: LogicalGraph, name: str) -> ModelContainer:
    """Transform LogicalNode and LogicalEdge objects into a LatexNode hierarchy.

    Args:
        graph: The logical computation graph to transcode.
        name: The identifier name of the resulting model container.

    Returns:
        ModelContainer: Root node representing the LaTeX model container AST.
    """
    state_registry = dict(graph.nodes)
    children: list[LatexNode] = self._build_memory_nodes(state_registry)

    input_node = next((n for n in graph.nodes.values() if n.op_type == "Input"), None)
    input_name = "input"
    children.append(InputNode(name=input_name, shape="[_]"))
    id_map: dict[str, str] = {input_node.id if input_node else "input": input_name}

    output_node = next((n for n in graph.nodes.values() if n.op_type == "Output"), None)
    visited_ops: set[str] = set()

    all_edges = list(graph.edges)
    if isinstance(graph, _PendingEdgesProtocol) and isinstance(graph._pending_edges, list):
      all_edges.extend(graph._pending_edges)

    for edge in all_edges:
      target_id = edge.target
      if target_id == "output" or (output_node and target_id == output_node.id) or target_id in visited_ops:
        continue

      is_stateful = (target_id in state_registry) and not target_id.startswith("func_")
      node_data = graph.nodes.get(target_id)
      op_type = str(node_data.op_type) if node_data else "Unknown"
      step_id = f"op_{target_id}"
      id_map[target_id] = step_id
      args = [id_map.get(edge.source, f"op_{edge.source}")]

      if is_stateful:
        children.append(StateOpNode(step_id, target_id, args, "[_]"))
      else:
        children.append(self._build_compute_node(step_id, op_type, args, node_data))
      visited_ops.add(target_id)

    if visited_ops:
      children.append(self._build_return_node(graph, id_map))

    return ModelContainer(name, children)
