"""
LaTeX DSL Emitter.

Converts Python Source -> Logical Graph -> LaTeX CST -> .tex string.
"""

from typing import List, Dict, Optional, Set
import libcst as cst
from ml_switcheroo.core.tikz.analyser import GraphExtractor, LogicalGraph
from ml_switcheroo.core.latex.nodes import (
  ModelContainer,
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
  LatexNode,
)


class LatexEmitter:
  def __init__(self) -> None:
    self.extractor = GraphExtractor()

  def emit(self, code: str, model_name: str = "GeneratedNet") -> str:
    """
    Converts Python source code to MIDL LaTeX.

    Args:
        code: The input Python source code.
        model_name: Default name for the model environment if one cannot be extracted.

    Returns:
        str: The LaTeX source string.
    """
    try:
      tree = cst.parse_module(code)
    except cst.ParserSyntaxError as e:
      return f"% Error parsing Python source: {e}"

    tree.visit(self.extractor)
    graph = self.extractor.graph

    # Priority Logic:
    # 1. If user provided a specific non-default name argument, use it (Override).
    # 2. Else if extractor found a class name, use it.
    # 3. Else fallback to "GeneratedNet".

    if model_name != "GeneratedNet":
      final_name = model_name
    elif self.extractor.model_name != "GeneratedNet":
      final_name = self.extractor.model_name
    else:
      final_name = "GeneratedNet"

    container = self._transcode_graph(graph, final_name)

    return self._wrap_document(container.to_latex())

  def _wrap_document(self, content: str) -> str:
    # Inject instructional header for users lacking the custom package
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
      r"\documentclass[tikz, border=10pt, landscape]{standalone}"
      "\n"
      r"\usepackage{midl}"
      "\n"
      r"\begin{document}"
      "\n"
    )
    footer = "\n" r"\end{document}"
    return comment_block + header + content + footer

  def _transcode_graph(self, graph: LogicalGraph, name: str) -> ModelContainer:
    children: List[LatexNode] = []
    state_registry = self.extractor.layer_registry

    # Memory (Attributes): State definitions
    # Filter out Input/Output and Ad-Hoc Functional Nodes
    for node_id, node in sorted(state_registry.items()):
      if node.kind in ["Input", "Output"]:
        continue
      if node_id.startswith("func_"):
        continue

      config = node.metadata.copy()
      mem = MemoryNode(node_id=node_id, op_type=node.kind, config=config)
      children.append(mem)

    # Handle Input
    input_node = next((n for n in graph.nodes if n.kind == "Input"), None)
    # Ensure input node uses standard ID "input" for LaTeX compliance if not explicit
    # The analyser forces node ID to "input", so this aligns.
    input_name = "input"
    children.append(InputNode(name=input_name, shape="[_]"))

    # Track mapping from logical node ID to Latex Step ID (op_id)
    # Initialize with input
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

      # Determine if target is stateful (Layer) or functional (Op)
      # It is stateful if it's in registry AND not marked as functional
      is_stateful = (target_id in state_registry) and not target_id.startswith("func_")

      node_data = next((n for n in graph.nodes if n.id == target_id), None)
      op_type = node_data.kind if node_data else "Unknown"
      step_id = f"op_{target_id}"

      # Register validity
      id_map[target_id] = step_id

      # Resolve source argument (Logic: use step ID if available, else usage raw ID fallback)
      # This ensures \Op{...}{...}{op_prev} linking
      arg_ref = id_map.get(source_id, f"op_{source_id}")
      args = [arg_ref]

      if is_stateful:
        children.append(StateOpNode(step_id, target_id, args, "[_]"))
      else:
        meta_args = []
        if node_data:
          for k, v in node_data.metadata.items():
            if k.startswith("arg"):
              meta_args.append(v)
            else:
              meta_args.append(f"{k}={v}")
        final_args = args + meta_args

        # Cleanup Type Name (e.g. func_relu -> Relu, F.relu -> Relu)
        clean_type = op_type
        if clean_type.startswith("func_"):
          # Remove prefix 'func_'
          clean_type = clean_type[5:]
        elif "." in clean_type:
          # Handle full paths like torch.nn.functional.relu -> relu
          # or F.relu -> relu
          clean_type = clean_type.split(".")[-1]

        # Capitalize for display (relu -> Relu)
        clean_type = clean_type.capitalize()

        children.append(ComputeNode(step_id, clean_type, final_args, "[_]"))

      visited_ops.add(target_id)

    if visited_ops:
      # Find what feeds Output to generate final Return
      sources_to_output = [e.source for e in graph.edges if e.target == "output" or e.target == "Output"]
      if sources_to_output:
        # Resolve the last op
        final_src = sources_to_output[0]
        final_ref = id_map.get(final_src, f"op_{final_src}")
        children.append(ReturnNode(target_id=final_ref))
      else:
        children.append(ReturnNode(target_id="last_step"))

    return ModelContainer(name, children)
