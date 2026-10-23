"""LaTeX visualizer backend for LogicalGraphs.

This module provides the LatexBackend, which compiles a LogicalGraph into
standalone LaTeX documents using the Machine Intelligence Definition Language (MIDL)
package format, suitable for high-quality PDF rendering of neural architectures.
"""

import typing

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
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

  def _transcode_graph(self, graph: LogicalGraph, name: str) -> ModelContainer:
    """Transform LogicalNode and LogicalEdge objects into a LatexNode hierarchy.

    Args:
        graph: The logical computation graph to transcode.
        name: The identifier name of the resulting model container.

    Returns:
        ModelContainer: Root node representing the LaTeX model container AST.
    """
    children: list[LatexNode] = []
    state_registry = dict(graph.nodes)

    for node_id, node in sorted(state_registry.items()):
      op_type = str(node.op_type)
      if op_type in ["Input", "Output"]:
        continue
      if node_id.startswith("func_"):
        continue
      config = {str(k): str(v) for k, v in node.attributes.items()}
      mem = MemoryNode(node_id=node_id, op_type=op_type, config=config)
      children.append(mem)

    input_node = next(
      (n for n in graph.nodes.values() if n.op_type == "Input"),
      None,
    )
    input_name = "input"
    children.append(InputNode(name=input_name, shape="[_]"))

    id_map = {}
    if input_node:
      id_map[input_node.id] = input_name
    else:
      id_map["input"] = input_name

    output_node = next(
      (n for n in graph.nodes.values() if n.op_type == "Output"),
      None,
    )
    visited_ops = set()

    all_edges = list(graph.edges)
    pending = getattr(graph, "_pending_edges", [])
    if pending:
      all_edges.extend(pending)

    for edge in all_edges:
      target_id = edge.target
      source_id = edge.source
      if target_id == "output" or (output_node and target_id == output_node.id):
        continue
      if target_id in visited_ops:
        continue

      is_stateful = (target_id in state_registry) and not target_id.startswith("func_")
      node_data = graph.nodes.get(target_id)
      op_type = str(node_data.op_type) if node_data else "Unknown"
      step_id = f"op_{target_id}"
      id_map[target_id] = step_id
      arg_ref = id_map.get(source_id, f"op_{source_id}")
      args = [arg_ref]

      if is_stateful:
        children.append(StateOpNode(step_id, target_id, args, "[_]"))
      else:
        meta_args = []
        if node_data:
          for k, v in node_data.attributes.items():
            val_str = str(v)
            if k.startswith("arg"):
              meta_args.append(val_str)
            else:
              meta_args.append(f"{k}={val_str}")
        final_args = args + meta_args
        clean_type = op_type
        if clean_type.startswith("func_"):
          clean_type = clean_type[5:]
        elif "." in clean_type:
          clean_type = clean_type.split(".")[-1]
        clean_type = clean_type.capitalize()
        children.append(ComputeNode(step_id, clean_type, final_args, "[_]"))

      visited_ops.add(target_id)

    if visited_ops:
      sources_to_output = [e.source for e in graph.edges if e.target == "output" or e.target == "Output"]
      if sources_to_output:
        final_src = sources_to_output[0]
        final_ref = id_map.get(final_src, f"op_{final_src}")
        children.append(ReturnNode(target_id=final_ref))
      else:
        children.append(ReturnNode(target_id="last_step"))

    return ModelContainer(name, children)
