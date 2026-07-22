"""Docstring."""

from typing import Any, Optional, List
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.latex.nodes import (
  LatexNode,
  ModelContainer,
  InputNode,
  ReturnNode,
  ComputeNode,
  MemoryNode,
  StateOpNode,
)


class LatexBackend(CompilerBackend):
  """Transforms Logical Graphs into MIDL LaTeX format."""

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Execute implementation detail."""
    pass

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles graph to LaTeX."""
    name = graph.name or "GeneratedNet"
    container = self._transcode_graph(graph, name)
    return self._wrap_document(container.to_latex())

  def _wrap_document(self, content: str) -> str:
    """Wraps content in Latex standalone document."""
    comment_block = (
      r"% ------------------------------------------------------------------"
      "\n"
      r"% [Requirement] midl.sty"
      "\n"
      r"% This document uses the Machine Intelligence Definition Language."
      "\n"
      r"% Ensure 'midl.sty' is in your LaTeX path."
      "\n"
      r"% ------------------------------------------------------------------"
      "\n"
    )
    header = (
      r"\documentclass[tikz, border=10pt, landscape]{standalone}" "\n" r"\usepackage{midl}" "\n" r"\begin{document}" "\n"
    )
    footer = "\n" + r"\end{document}"
    return comment_block + header + content + footer

  def _transcode_graph(self, graph: LogicalGraph, name: str) -> ModelContainer:
    """Transforms LogicalNodes to LatexNode hierarchy."""
    children: List[LatexNode] = []
    # Reconstruct registry as graph nodes list
    state_registry = {n.id: n for n in graph.nodes}

    # Memory Logic
    for node_id, node in sorted(state_registry.items()):
      if node.kind in ["Input", "Output"]:
        continue
      if node_id.startswith("func_"):
        continue
      config = node.metadata.copy()
      mem = MemoryNode(node_id=node_id, op_type=node.kind, config=config)
      children.append(mem)

    input_node = next((n for n in graph.nodes if n.kind == "Input"), None)
    input_name = "input"
    children.append(InputNode(name=input_name, shape="[_]"))

    id_map = {}
    if input_node:
      id_map[input_node.id] = input_name
    else:
      id_map["input"] = input_name

    output_node = next((n for n in graph.nodes if n.kind == "Output"), None)
    visited_ops = set()

    for edge in graph.edges:
      target_id = edge.target
      source_id = edge.source
      if target_id == "output" or (output_node and target_id == output_node.id):
        continue
      if target_id in visited_ops:
        continue

      is_stateful = (target_id in state_registry) and not target_id.startswith("func_")
      node_data = next((n for n in graph.nodes if n.id == target_id), None)
      op_type = node_data.kind if node_data else "Unknown"
      step_id = f"op_{target_id}"
      id_map[target_id] = step_id
      arg_ref = id_map.get(source_id, f"op_{source_id}")
      args = [arg_ref]

      if is_stateful:
        children.append(StateOpNode(step_id, target_id, args, "[_]"))
      else:
        meta_args = []
        if node_data:  # pragma: no cover
          for k, v in node_data.metadata.items():
            if k.startswith("arg"):
              meta_args.append(v)
            else:
              meta_args.append(f"{k}={v}")
        final_args = args + meta_args
        clean_type = op_type
        if clean_type.startswith("func_"):
          clean_type = clean_type[5:]
        elif "." in clean_type:  # pragma: no cover
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
