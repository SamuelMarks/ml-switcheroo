"""
HTML DSL Semantic Nodes.

Defines the structure for the visual elements used in the HTML/SVG DSL:
- GridBox: Represents a layer (Red), operation (Blue), or data shape (Green).
- SvgArrow: Represents data flow connections.
- HtmlDocument: The root container holding CSS and body content.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HtmlNode:
  """
  Abstract base class for all HTML DSL elements.
  """

  def to_html(self) -> str:
    """
    Render the node and its children to an HTML string.

    Returns:
        str: The raw HTML content.

    Raises:
        NotImplementedError: If not implemented by subclass.
    """
    raise NotImplementedError


@dataclass
class SvgArrow(HtmlNode):
  """
  Represents an SVG connection line between grid cells.

  Attributes:
      x1 (int): Start X coordinate.
      y1 (int): Start Y coordinate.
      x2 (int): End X coordinate.
      y2 (int): End Y coordinate.
      style_class (str): CSS class for stroke color/style (e.g. 's-red', 's-blue').
      marker_end (str): Marker ID for arrowhead (e.g. 'url(#mr)').
      parent_style (str): CSS string for absolute positioning of the container SVG.
                          Example: 'left:100%; bottom:20px;'
  """

  x1: int
  y1: int
  x2: int
  y2: int
  style_class: str
  marker_end: str
  parent_style: str

  def to_html(self) -> str:
    """
    Renders the arrow as an absolute SVG element.
    Adds 'sw-arrow' class for scoped styling.
    """
    return f""" 
    <svg class="sw-arrow" style="{self.parent_style}">
      <line x1="{self.x1}" y1="{self.y1}" x2="{self.x2}" y2="{self.y2}" class="{self.style_class}" marker-end="{self.marker_end}" />
    </svg>"""


@dataclass
class GridBox(HtmlNode):
  """
  Represents a content box positioned within the CSS Grid.

  Attributes:
      row (int): CSS Grid row index.
      col (int): CSS Grid column index.
      css_class (str): Visual styling classes (e.g. 'box r', 'circ').
      header_text (str): Main label (e.g. 'Conv2d', 'Return').
      code_text (Optional[str]): Parameter details formatted as code.
      body_text (Optional[str]): Additional text (e.g. Shape info like '[B, 32]').
      arrows (List[SvgArrow]): List of outgoing or decorative arrows attached to this box.
      z_index (Optional[int]): Explicit stack order.
  """

  row: int
  col: int
  css_class: str
  header_text: str
  code_text: Optional[str] = None
  body_text: Optional[str] = None
  arrows: List[SvgArrow] = field(default_factory=list)
  z_index: Optional[int] = None

  def to_html(self) -> str:
    """
    Renders the grid cell div, its content, and attached arrows.
    """
    style = f"grid-row:{self.row}; grid-column:{self.col};"
    if self.z_index is not None:
      style += f" z-index:{self.z_index};"

    content = []

    # Handle 'circ' class special layout (flex centered single text)
    if "circ" in self.css_class:
      content.append(f"{self.header_text}")
    else:
      content.append(f'<span class="header-txt">{self.header_text}</span>')

    if self.code_text:
      # Wrap code text
      content.append(f"<code>{self.code_text}</code>")

    if self.body_text:
      content.append(self.body_text.strip())

    # Render arrows inside the box div to allow relative positioning
    arrow_html = "".join(a.to_html() for a in self.arrows)

    return f""" 
  <div class="{self.css_class}" style="{style}">
    {"".join(content)} 
    {arrow_html} 
  </div>"""


@dataclass
class HtmlDocument(HtmlNode):
  """
  Root container for the generated HTML.

  Attributes:
      model_name (str): Title displayed in H1.
      children (List[GridBox]): List of grid elements.
  """

  model_name: str
  children: List[GridBox]

  # CSS Definition
  # Updates:
  # 1. Scoped .grid -> .sw-grid to avoid conflicts.
  # 2. Replaced global 'svg' selector with '.sw-arrow'.
  # 3. Changed .col-mid-bg to 'grid-row: 1 / -1' for full height.
  _CSS = """ 
  .sw-grid { 
    display: grid; 
    grid-template-columns: 1fr 200px 1fr; 
    gap: 40px; 
    position: relative; 
    max-width: 950px; 
    font-family: sans-serif; 
    font-size: 14px; 
    background-color: #fcfcfc; 
    padding: 20px; 
  } 

  /* Middle Column Borders (Background) */ 
  .col-mid-bg { 
    grid-column: 2; 
    /* Span from row 1 to the end */ 
    grid-row: 1 / -1; 
    border-left: 2px dotted #bbb; 
    border-right: 2px dotted #bbb; 
    z-index: 0; 
    pointer-events: none; 
  } 

  /* HEADERS */ 
  .sw-grid h3 { margin: 0; font-size: 16px; text-decoration: underline; white-space: nowrap; align-self: end; padding-bottom: 10px; color: #333; } 

  /* BOX STYLES */ 
  .sw-grid .box { 
    border: 2px solid; 
    padding: 8px 12px; 
    border-radius: 6px; 
    background: white; 
    position: relative; 
    display: flex; 
    flex-direction: column; 
    justify-content: center; 
    box-sizing: border-box; 
    height: 100%; 
    z-index: 2; 
  } 

  .sw-grid .header-txt { font-weight: bold; color: black; margin-bottom: 4px; display: block; } 
  .sw-grid code { font-family: monospace; font-size: 12px; color: #444; display: block; background: rgba(0,0,0,0.05); padding: 2px; border-radius: 3px; } 

  /* COLORS & SPECIFICS */ 
  .sw-grid .r { border-color: #d00; background: #ffecec; } 
  .sw-grid .b { border-color: #00d; background: #ecf0ff; width: 90%; justify-self: center; } 
  .sw-grid .g { border-color: #080; background: #ecffec; } 

  .sw-grid .circ { 
    width: 60px; height: 60px; 
    border-radius: 50%; 
    background: darkblue; color: white; 
    display: flex; align-items: center; justify-content: center; 
    font-weight: bold; 
    justify-self: center; align-self: center; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.2); 
    z-index: 2; 
  } 

  /* SVG Overlays */ 
  .sw-arrow { 
    position: absolute; 
    overflow: visible; 
    pointer-events: none; 
    z-index: 10; 
    width: 1px; 
    height: 1px; 
  } 

  .sw-grid .s-red   { stroke: #d00; stroke-width: 2; stroke-dasharray: 4; fill: none; } 
  .sw-grid .s-blue  { stroke: #00d; stroke-width: 2; fill: none; } 
  .sw-grid .s-green { stroke: #080; stroke-width: 2; stroke-dasharray: 4; fill: none; } 
"""

  def to_html(self) -> str:
    """
    Renders the complete HTML document.
    Notes:
    - Removed 'visibility:hidden' from markers block to enable correct rendering in some browsers
      when injected via innerHTML. Use 0 sizes and pointer-events logic instead.
    """
    # Determine strict grid height
    repeat_count = 0
    if self.children:
      max_used = max(c.row for c in self.children)
      repeat_count = max(0, max_used - 1)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
{self._CSS} 
  /* Explicit Row Heights */ 
  .sw-grid {{ 
    grid-template-rows: 30px repeat({repeat_count}, 80px); 
  }} 
</style>
</head>
<body>

<!-- MARKERS: Must be visible to DOM engine but hidden from layout -->
<svg style="width:0;height:0;position:absolute;overflow:hidden;" aria-hidden="true">
  <defs>
    <marker id="mr" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#d00"/></marker>
    <marker id="mb" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#00d"/></marker>
    <marker id="mg" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#080"/></marker>
  </defs>
</svg>

<h3>Model: {self.model_name}</h3>

<div class="sw-grid">
  <div class="col-mid-bg"></div>

  <!-- HEADERS -->
  <div style="grid-row:1; grid-column:1;"><h3>Memory (Init)</h3></div>
  <div style="grid-row:1; grid-column:2; text-align:center;"><h3>Computer (forward)</h3></div>
  <div style="grid-row:1; grid-column:3;"><h3>Data (shape)</h3></div>

  {"".join(c.to_html() for c in self.children)} 
</div>

</body>
</html>"""
