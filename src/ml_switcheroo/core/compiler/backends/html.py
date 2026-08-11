"""HTML Visualizer Backend."""

from typing import Any, Dict, Optional, List
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, topological_sort
from ml_switcheroo.core.html.nodes import SvgArrow, GridBox, HtmlDocument


class HtmlBackend(CompilerBackend):
  """Orchestrates the conversion of Logical Graphs to the HTML visual DSL."""

  # Layout Constants
  ROW_HEIGHT = 80
  GAP_HEIGHT = 0

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize the HTML backend.

    Args:
        semantics: Unused semantics manager (for protocol compatibility).

    """
    pass

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles the graph into an HTML string document.

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
    """Format metadata dict into string.

    Args:
        metadata: Node metadata arguments.

    Returns:
        Formatted argument string.
    """
    parts = []
    for k, v in metadata.items():
      if k.startswith("arg_"):
        parts.append(str(v))
      else:
        parts.append(f"{k}={v}")
    return ", ".join(parts)

  def _is_stateful(self, node: LogicalNode) -> bool:
    """Determine if a node represents state (Red box) vs Op (Blue box).

    Args:
        node: The node to check.

    Returns:
        True if the node is stateful, False otherwise.
    """
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
    """Clean operation kind string.

    Args:
        kind: The original operation kind.

    Returns:
        The cleaned operation kind string.
    """
    if kind.startswith("func_"):
      kind = kind[5:]
    if "." in kind:
      kind = kind.split(".")[-1]
    return kind.capitalize()

  def _create_arrow(self, start_row: int, end_row: int, arrow_type: str = "seq") -> SvgArrow:
    """Factory for SvgArrows based on row distance.

    Args:
        start_row: The starting row index.
        end_row: The ending row index.
        arrow_type: Type of arrow to create ('def', 'data', 'seq').

    Returns:
        A generated SvgArrow.
    """
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

    return SvgArrow(x1=0, y1=0, x2=0, y2=0)

  def _layout_graph(self, graph: LogicalGraph) -> List[GridBox]:
    """Calculates grid positions for nodes.

    Args:
        graph: The logical graph.

    Returns:
        A list of GridBox elements representing the layout.
    """
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
        if last_blue_row != -1:  # pragma: no branch
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

    if last_blue_row != -1:  # pragma: no branch
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
