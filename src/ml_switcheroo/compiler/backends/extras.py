"""
Extra Backends (Standardized implementations for Visual/DSL Targets).

Provides CompilerBackend implementations for:
- TikZ (Visualization)
- HTML (Visualization)
- LaTeX DSL (MIDL)
- MLIR (Graph to MLIR Text)
- StableHLO (Graph to StableHLO-flavored MLIR Text)

These backends consume the LogicalGraph IR and emit target-specific textual representations.
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, topological_sort

# HTML Dependencies
from ml_switcheroo.core.html.nodes import HtmlDocument, GridBox, SvgArrow

# TikZ Dependencies
from ml_switcheroo.core.tikz.nodes import (
  TikzGraph,
  TikzNode,
  TikzEdge,
  TikzOption,
  TikzTable,
  TikzComment,
  TriviaNode,
)

# LaTeX Dependencies
from ml_switcheroo.core.latex.nodes import (
  ModelContainer,
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
  LatexNode,
)


class HtmlBackend(CompilerBackend):
  """
  Orchestrates the conversion of Logical Graphs to the HTML visual DSL.
  """

  # Layout Constants
  ROW_HEIGHT = 80
  GAP_HEIGHT = 0

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """
    Initialize the HTML backend.

    Args:
        semantics: Unused semantics manager (for protocol compatibility).
    """
    pass

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compiles the graph into an HTML string document.

    Args:
        graph: The logical graph.

    Returns:
        HTML string.
    """
    children = self._layout_graph(graph)
    model_name = graph.name if graph.name and graph.name != "GeneratedNet" else "ConvNet"
    doc = HtmlDocument(model_name=model_name, children=children)
    return doc.to_html()

  def _format_args(self, metadata: Dict[str, str]) -> str:
    """Format metadata dict into string."""
    parts = []
    for k, v in metadata.items():
      if k.startswith("arg_"):
        parts.append(str(v))
      else:
        parts.append(f"{k}={v}")
    return ", ".join(parts)

  def _is_stateful(self, node: LogicalNode) -> bool:
    """Determine if a node represents state (Red box) vs Op (Blue box)."""
    if node.kind in ["Input", "Output"]:
      return False
    if node.id.startswith("func_"):
      return False
    if node.kind.startswith("func_"):
      return False
    # Heuristic: Upper case kinds are layers/stateful
    if node.kind and node.kind[0].isupper():
      return True
    return False

  def _clean_kind(self, kind: str) -> str:
    """Clean operation kind string."""
    if kind.startswith("func_"):
      kind = kind[5:]
    if "." in kind:
      kind = kind.split(".")[-1]
    return kind.capitalize()

  def _create_arrow(self, start_row: int, end_row: int, arrow_type: str = "seq") -> SvgArrow:
    """Factory for SvgArrows based on row distance."""
    if arrow_type == "def":
      # Red -> Blue (Right and Down)
      return SvgArrow(
        x1=0,
        y1=0,
        x2=60,
        y2=80,
        style_class="s-red",
        marker_end="url(#mr)",
        parent_style="left:100%; top:60px;",
      )

    if arrow_type == "data":
      # Blue -> Green (Straight Right)
      return SvgArrow(
        x1=0,
        y1=0,
        x2=60,
        y2=0,
        style_class="s-green",
        marker_end="url(#mg)",
        parent_style="left:100%; top:40px;",
      )

    if arrow_type == "seq":
      # Blue -> Blue (Down)
      row_delta = end_row - start_row
      # Formula: 50 + (delta-1)*(box+gap) = 50 + (d-1)*120
      # Assuming row height + gap ~ 120 pixels? Based on ROW_HEIGHT=80.
      # Using derived constant from visual testing.
      y_len = 50 + (row_delta - 1) * 120

      return SvgArrow(
        x1=0,
        y1=0,
        x2=0,
        y2=y_len,
        style_class="s-blue",
        marker_end="url(#mb)",
        parent_style="top:80px; left:50%;",
      )

    return SvgArrow(0, 0, 0, 0, "", "", "")

  def _layout_graph(self, graph: LogicalGraph) -> List[GridBox]:
    """Calculates grid positions for nodes."""
    boxes = []
    ordered = topological_sort(graph)
    current_row = 2  # Row 1 is Headers
    last_blue_row = -1  # Track last operation box

    flow_nodes = [n for n in ordered if n.kind != "Input" and n.kind != "Output"]

    if not flow_nodes:
      # Render empty? Or input only?
      return []

    current_z = 1000

    for i, node in enumerate(flow_nodes):
      is_stateful = self._is_stateful(node)
      op_row = current_row

      if is_stateful:
        # 1. Attribute Box (Red)
        disp_kind = self._clean_kind(node.kind)
        mem_box = GridBox(
          row=current_row,
          col=1,
          css_class="box r",
          header_text=f"{node.id}: {disp_kind}",
          code_text=self._format_args(node.metadata),
          z_index=current_z,
        )
        current_z -= 1
        mem_box.arrows.append(self._create_arrow(0, 0, "def"))
        boxes.append(mem_box)
        op_row = current_row + 1

      # 2. Operation Box (Blue)
      if is_stateful:
        op_label = f"Call ({node.id})"
        op_args = "args: x"
      else:
        op_label = self._clean_kind(node.kind)
        fmt_args = self._format_args(node.metadata)
        op_args = f"args: {fmt_args}" if fmt_args else "args: x"

      op_box = GridBox(
        row=op_row,
        col=2,
        css_class="box b",
        header_text=op_label,
        code_text=op_args,
        z_index=current_z,
      )
      current_z -= 1

      # Sequential Arrow (Blue)
      if i == 0:
        op_box.arrows.append(
          SvgArrow(
            x1=0,
            y1=0,
            x2=0,
            y2=50,
            style_class="s-blue",
            marker_end="url(#mb)",
            parent_style="top:-52px; left:50%;",
          )
        )
      else:
        if last_blue_row != -1:
          arrow = self._create_arrow(last_blue_row, op_row, "seq")
          for b in boxes:
            if b.row == last_blue_row and "box b" in b.css_class:
              b.arrows.append(arrow)
              break

      op_box.arrows.append(self._create_arrow(0, 0, "data"))
      boxes.append(op_box)
      last_blue_row = op_row

      # 3. Data Box (Green)
      data_box = GridBox(
        row=op_row,
        col=3,
        css_class="box g",
        header_text=f"out_{node.id}",
        body_text="[_]",
        z_index=current_z,
      )
      current_z -= 1
      boxes.append(data_box)

      step = 2 if is_stateful else 1
      current_row += step

    # 4. Return Bubble
    return_row = current_row
    arrow = self._create_arrow(last_blue_row, return_row, "seq")

    if last_blue_row != -1:
      for b in boxes:
        if b.row == last_blue_row and "box b" in b.css_class:
          b.arrows.append(arrow)
          break

    return_circle = GridBox(
      row=return_row,
      col=2,
      css_class="circ",
      header_text="Return",
      z_index=current_z,
    )
    boxes.append(return_circle)

    return boxes


class TikzBackend(CompilerBackend):
  """
  Orchestrates the conversion of a LogicalGraph to TikZ source code.
  Verified to use Rank-Based Layout.
  """

  def __init__(self, semantics: Optional[Any] = None, y_spacing: float = 2.5, x_spacing: float = 3.0) -> None:
    """
    Initialize TikZ backend.

    Args:
        semantics: Unused.
        y_spacing: Vertical space between ranks.
        x_spacing: Horizontal space between nodes.
    """
    self.y_spacing = y_spacing
    self.x_spacing = x_spacing

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compile graph to TikZ LaTeX code.
    """
    positions = self._calculate_layout(graph)
    cst_nodes = []
    cst_nodes.append(TikzComment("Generated by ml-switcheroo"))
    cst_nodes.append(TriviaNode("\n"))

    for node in graph.nodes:
      pos = positions.get(node.id, (0, 0))
      tikz_node = self._create_tikz_node(node, pos[0], pos[1])
      cst_nodes.append(tikz_node)

    cst_nodes.append(TriviaNode("\n"))

    for edge in graph.edges:
      tikz_edge = self._create_tikz_edge(edge)
      cst_nodes.append(tikz_edge)

    options = [
      TikzOption("node distance", "2cm"),
      TikzOption("auto"),
      TikzOption(">=stealth"),
    ]
    root = TikzGraph(children=cst_nodes, options=options)
    return root.to_text()

  def _sanitize(self, text: str) -> str:
    """Escape special LaTeX characters."""
    return str(text).replace("_", r"\_")

  def _calculate_layout(self, graph: LogicalGraph) -> Dict[str, Tuple[float, float]]:
    """Determine X,Y coordinates for nodes based on topological rank."""
    if not graph.nodes:
      return {}

    adj = defaultdict(list)
    in_degree = defaultdict(int)
    for n in graph.nodes:
      in_degree[n.id] = 0

    for edge in graph.edges:
      adj[edge.source].append(edge.target)
      in_degree[edge.target] += 1

    queue = deque([n.id for n in graph.nodes if in_degree[n.id] == 0])
    ranks = {}
    processed = set()

    if not queue and graph.nodes:
      first = graph.nodes[0].id
      queue.append(first)
      ranks[first] = 0
    else:
      for root_id in queue:
        ranks[root_id] = 0

    max_rank = 0
    while queue:
      curr = queue.popleft()
      if curr in processed:
        continue
      processed.add(curr)
      curr_rank = ranks.get(curr, 0)
      max_rank = max(max_rank, curr_rank)

      for neighbor in adj[curr]:
        if neighbor not in ranks or ranks[neighbor] < curr_rank + 1:
          ranks[neighbor] = curr_rank + 1
          queue.append(neighbor)

    for n in graph.nodes:
      if n.id not in ranks:
        max_rank += 1
        ranks[n.id] = max_rank

    rank_groups = defaultdict(list)
    for node_id, r in ranks.items():
      rank_groups[r].append(node_id)

    positions = {}
    for r, nodes in rank_groups.items():
      y = -r * self.y_spacing
      count = len(nodes)
      start_x = -((count - 1) * self.x_spacing) / 2
      for i, node_id in enumerate(nodes):
        x = start_x + (i * self.x_spacing)
        positions[node_id] = (x, y)
    return positions

  def _create_tikz_node(self, node: LogicalNode, x: float, y: float) -> TikzNode:
    """Creates a TikzNode AST object."""
    options = [
      TikzOption("draw"),
      TikzOption("rectangle"),
      TikzOption("rounded corners"),
      TikzOption("align", "center"),
    ]
    if node.kind == "Input":
      options.append(TikzOption("fill", "green!10"))
    elif node.kind == "Output":
      options.append(TikzOption("fill", "red!10"))
    else:
      options.append(TikzOption("fill", "blue!5"))

    sanitized_kind = self._sanitize(node.kind)
    rows = [[rf"\textbf{{{sanitized_kind}}}"]]
    sanitized_id = self._sanitize(node.id)
    rows.append([rf"\textit{{{sanitized_id}}}"])

    for k, v in node.metadata.items():
      clean_k = self._sanitize(k)
      clean_v = self._sanitize(str(v)[:20])
      rows.append([f"{clean_k}: {clean_v}"])

    content_table = TikzTable(rows=rows, align="c")

    return TikzNode(
      node_id=node.id,
      x=x,
      y=y,
      content=content_table,
      options=options,
    )

  def _create_tikz_edge(self, edge: LogicalEdge) -> TikzEdge:
    """Creates a TikzEdge AST object."""
    return TikzEdge(
      source_id=edge.source,
      target_id=edge.target,
      connector="--",
      options=[TikzOption("->"), TikzOption("thick")],
    )


class LatexBackend(CompilerBackend):
  """
  Transforms Logical Graphs into MIDL LaTeX format.
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
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
      r"\documentclass[tikz, border=10pt, landscape]{standalone}"
      "\n"
      r"\usepackage{midl}"
      "\n"
      r"\begin{document}"
      "\n"
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
        if node_data:
          for k, v in node_data.metadata.items():
            if k.startswith("arg"):
              meta_args.append(v)
            else:
              meta_args.append(f"{k}={v}")
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


class MlirBackend(CompilerBackend):
  """
  Back-end for generating MLIR text from a LogicalGraph.

  This implementation provides a direct Graph-to-MLIR conversion path for
  scenarios bypassing the high-level Python CST Rewriter (e.g. source is RDNA).
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize."""
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compiles the LogicalGraph into an MLIR string representation.

    It emits a simple `module` structure with a single block, mapping each
    logical node to an `sw.op` operation. Inputs are generated as
    `sw.constant` if metadata values exist, or `sw.op {type="Input"}`.

    Args:
        graph: The logical graph IR.

    Returns:
        str: The generated MLIR code.
    """
    lines = ["// Graph -> MLIR compilation output"]
    lines.append("module {")
    lines.append("  func.func @main() {")

    for node in graph.nodes:
      if node.kind == "Input":
        val = node.metadata.get("value", "1")
        # Try to determine type
        # If value is integer-like, cast to i32
        if str(val).isdigit():
          lines.append(f'    %{node.id} = "sw.constant"() {{value = {val}}} : () -> i32')
        else:
          # Treat input as argument or placeholder constant
          lines.append(f'    %{node.id} = "sw.op"() {{type = "Input"}} : () -> !sw.unknown')
      elif node.kind == "Output":
        # Sink node, often no output or uses return logic
        # Find source
        # Graph edges handling is simplistic here: just list nodes
        lines.append(f'    "sw.return"() : () -> ()')
      else:
        # Generic Op
        # Construct args string from metadata
        attrs = []
        for k, v in node.metadata.items():
          attrs.append(f'{k} = "{v}"')
        attr_str = ", ".join(attrs)
        attr_block = f" {{{attr_str}}}" if attrs else ""
        lines.append(f'    %{node.id} = "sw.op"() {{type = "{node.kind}"}}{attr_block} : () -> !sw.unknown')

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


class StableHloBackend(CompilerBackend):
  """
  Back-end for generating StableHLO text from a LogicalGraph.

  This implementation provides a direct Graph-to-StableHLO conversion path
  used when an ISA is the source format (e.g. SASS -> StableHLO).
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize."""
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """
    Compiles the graph to StableHLO-flavored MLIR.

    Args:
        graph: The logical graph.

    Returns:
        str: MLIR code string using stablehlo dialect.
    """
    lines = ["// Graph -> StableHLO compilation output"]
    lines.append("module {")
    lines.append("  func.func @main() {")

    for node in graph.nodes:
      if node.kind == "Input":
        lines.append(f"    %{node.id} = stablehlo.constant dense<0.0> : tensor<f32>")
      elif node.kind == "Output":
        lines.append("    return")
      else:
        # Attempt simpler mapping
        op_name = node.kind.lower().split(".")[-1]
        lines.append(f"    %{node.id} = stablehlo.custom_call @{op_name}(...)")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
