"""HTML DSL Semantic Nodes.

Defines the structure for the visual elements used in the HTML/SVG DSL:
- GridBox: Represents a layer (Red), operation (Blue), or data shape (Green).
- SvgArrow: Represents data flow connections.
- HtmlDocument: The root container holding CSS and body content.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from ml_switcheroo.utils.formatting import StructuredFormatter, escape_html


@dataclass
class HtmlNode:
  """Abstract base class for all HTML DSL elements."""

  leading_trivia: str = ""
  trailing_trivia: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Render the node and its children to an HTML string with indentation.

    Args:
        indent_level: The indentation level.

    Returns:
        str: The structured HTML content.

    Raises:
        NotImplementedError: If not implemented by subclass.

    """
    raise NotImplementedError

  def to_html(self) -> str:
    """Convenience method to render HTML."""
    return self.emit(0)


@dataclass
class TextNode(HtmlNode):
  """Represents a raw text string in the DOM."""

  content: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the text content."""
    return f"{self.leading_trivia}{self.content}{self.trailing_trivia}"


@dataclass
class CommentNode(HtmlNode):
  """Represents an HTML comment <!-- ... -->."""

  content: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the HTML comment."""
    return f"{self.leading_trivia}<!--{self.content}-->{self.trailing_trivia}"


@dataclass
class AttributeNode(HtmlNode):
  """Represents an HTML tag attribute."""

  name: str = ""
  value: Optional[str] = None
  quote_style: str = '"'  # '"', "'", or ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the attribute."""
    if self.value is None:
      return f"{self.leading_trivia}{self.name}{self.trailing_trivia}"
    return f"{self.leading_trivia}{self.name}={self.quote_style}{self.value}{self.quote_style}{self.trailing_trivia}"


@dataclass
class TagNode(HtmlNode):
  """Represents an HTML Element."""

  name: str = ""
  attributes: List[AttributeNode] = field(default_factory=list)
  children: List[HtmlNode] = field(default_factory=list)
  self_closing: bool = False

  def emit(self, indent_level: int = 0) -> str:
    """Emit the HTML tag."""
    parts = [self.leading_trivia, f"<{self.name}"]
    for attr in self.attributes:
      if not attr.leading_trivia:
        parts.append(" ")
      parts.append(attr.emit())

    if self.self_closing:
      parts.append("/>")
      parts.append(self.trailing_trivia)
      return "".join(parts)

    parts.append(">")
    for child in self.children:
      parts.append(child.emit(indent_level))

    parts.append(f"</{self.name}>")
    parts.append(self.trailing_trivia)
    return "".join(parts)


@dataclass
class SvgArrow(HtmlNode):
  """Represents an SVG connection line between grid cells."""

  x1: int = 0
  y1: int = 0
  x2: int = 0
  y2: int = 0
  style_class: str = ""
  marker_end: str = ""
  parent_style: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Renders the arrow as an absolute SVG element."""
    fmt = StructuredFormatter()
    style = escape_html(self.parent_style)
    cls = escape_html(self.style_class)
    marker = escape_html(self.marker_end)

    fmt.add_line(f'<svg class="sw-arrow" style="{style}">', indent_level)
    fmt.add_line(
      f'<line x1="{self.x1}" y1="{self.y1}" x2="{self.x2}" y2="{self.y2}" class="{cls}" marker-end="{marker}" />',
      indent_level + 1,
    )
    fmt.add_line("</svg>", indent_level)
    return fmt.build()


@dataclass
class GridBox(HtmlNode):
  """Represents a content box positioned within the CSS Grid."""

  row: int = 0
  col: int = 0
  css_class: str = ""
  header_text: str = ""
  code_text: Optional[str] = None
  body_text: Optional[str] = None
  arrows: List[SvgArrow] = field(default_factory=list)
  z_index: Optional[int] = None

  def emit(self, indent_level: int = 0) -> str:
    """Renders the grid cell div, its content, and attached arrows."""
    fmt = StructuredFormatter()
    style = f"grid-row:{self.row}; grid-column:{self.col};"
    if self.z_index is not None:
      style += f" z-index:{self.z_index};"

    safe_cls = escape_html(self.css_class)
    safe_style = escape_html(style)

    fmt.add_line(f'<div class="{safe_cls}" style="{safe_style}">', indent_level)

    # Handle 'circ' class special layout (flex centered single text)
    safe_header = escape_html(self.header_text)
    if "circ" in self.css_class:
      fmt.add_line(f"{safe_header}", indent_level + 1)
    else:
      fmt.add_line(f'<span class="header-txt">{safe_header}</span>', indent_level + 1)

    if self.code_text:
      safe_code = escape_html(self.code_text)
      fmt.add_line(f"<code>{safe_code}</code>", indent_level + 1)

    if self.body_text:
      safe_body = escape_html(self.body_text.strip())
      fmt.add_line(f"{safe_body}", indent_level + 1)

    # Render arrows inside the box div to allow relative positioning
    for arrow in self.arrows:
      fmt.add_line(
        arrow.emit(indent_level + 1), 0
      )  # Emit already handles its own internal indent, but here we just pass the block

    fmt.add_line("</div>", indent_level)
    return fmt.build()


@dataclass
class HtmlDocument(HtmlNode):
  """Root container for the generated HTML."""

  model_name: str = ""
  children: List[Any] = field(default_factory=list)

  # CSS Definition
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

  def emit(self, indent_level: int = 0) -> str:
    """Renders the complete HTML document."""
    # Check if we are a pure CST HtmlDocument representation (children are TagNodes)
    # vs structured representation
    is_pure_cst = bool(self.children) and all(isinstance(c, (TagNode, TextNode, CommentNode)) for c in self.children)
    if is_pure_cst:
      parts = [self.leading_trivia]
      for child in self.children:
        parts.append(child.emit(indent_level))
      parts.append(self.trailing_trivia)
      return "".join(parts)

    fmt = StructuredFormatter()

    repeat_count = 0
    if self.children:
      max_used = max(getattr(c, "row", 0) for c in self.children if hasattr(c, "row"))
      repeat_count = max(0, max_used - 1)

    fmt.add_line("<!DOCTYPE html>", indent_level)
    fmt.add_line("<html>", indent_level)
    fmt.add_line("<head>", indent_level)
    fmt.add_line("<style>", indent_level)
    fmt.add_block(self._CSS.strip(), indent_level + 1)
    fmt.add_line("/* Explicit Row Heights */", indent_level + 1)
    fmt.add_line(".sw-grid {", indent_level + 1)
    fmt.add_line(f"  grid-template-rows: 30px repeat({repeat_count}, 80px);", indent_level + 1)
    fmt.add_line("}", indent_level + 1)
    fmt.add_line("</style>", indent_level)
    fmt.add_line("</head>", indent_level)
    fmt.add_line("<body>", indent_level)
    fmt.add_line("", indent_level)
    fmt.add_line("<!-- MARKERS: Must be visible to DOM engine but hidden from layout -->", indent_level)
    fmt.add_line('<svg style="width:0;height:0;position:absolute;overflow:hidden;" aria-hidden="true">', indent_level)
    fmt.add_line("  <defs>", indent_level)
    fmt.add_line(
      '    <marker id="mr" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#d00"/></marker>',
      indent_level,
    )
    fmt.add_line(
      '    <marker id="mb" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#00d"/></marker>',
      indent_level,
    )
    fmt.add_line(
      '    <marker id="mg" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#080"/></marker>',
      indent_level,
    )
    fmt.add_line("  </defs>", indent_level)
    fmt.add_line("</svg>", indent_level)
    fmt.add_line("", indent_level)

    safe_name = escape_html(self.model_name)
    fmt.add_line(f"<h3>Model: {safe_name}</h3>", indent_level)
    fmt.add_line("", indent_level)
    fmt.add_line('<div class="sw-grid">', indent_level)
    fmt.add_line('  <div class="col-mid-bg"></div>', indent_level)
    fmt.add_line("", indent_level)
    fmt.add_line("  <!-- HEADERS -->", indent_level)
    fmt.add_line('  <div style="grid-row:1; grid-column:1;"><h3>Memory (Init)</h3></div>', indent_level)
    fmt.add_line(
      '  <div style="grid-row:1; grid-column:2; text-align:center;"><h3>Computer (forward)</h3></div>', indent_level
    )
    fmt.add_line('  <div style="grid-row:1; grid-column:3;"><h3>Data (shape)</h3></div>', indent_level)
    fmt.add_line("", indent_level)

    for child in self.children:
      fmt.add_line(child.emit(indent_level + 1), 0)

    fmt.add_line("</div>", indent_level)
    fmt.add_line("", indent_level)
    fmt.add_line("</body>", indent_level)
    fmt.add_line("</html>", indent_level)
    return fmt.build()
