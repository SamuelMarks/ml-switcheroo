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
  """Abstract base class for all HTML DSL elements.

  Attributes:
      leading_trivia: Whitespace or comments preceding this node.
      trailing_trivia: Whitespace or comments succeeding this node.
  """

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
    """Convenience method to render HTML.

    Returns:
        str: The rendered HTML string.
    """
    return self.emit(0)


@dataclass
class TextNode(HtmlNode):
  """Represents a raw text string in the DOM.

  Attributes:
      content: The raw text content.
  """

  content: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the text content.

    Args:
        indent_level: The current indentation level (unused in this node).

    Returns:
        str: The raw text content with leading and trailing trivia.
    """
    return f"{self.leading_trivia}{self.content}{self.trailing_trivia}"


@dataclass
class CommentNode(HtmlNode):
  """Represents an HTML comment <!-- ... -->.

  Attributes:
      content: The content within the HTML comment.
  """

  content: str = ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the HTML comment.

    Args:
        indent_level: The current indentation level (unused in this node).

    Returns:
        str: The formatted HTML comment with leading and trailing trivia.
    """
    return f"{self.leading_trivia}<!--{self.content}-->{self.trailing_trivia}"


@dataclass
class AttributeNode(HtmlNode):
  """Represents an HTML tag attribute.

  Attributes:
      name: The name of the HTML attribute.
      value: The value of the attribute, or None if it has no value.
      quote_style: The quote style to use ('"', "'", or empty).
  """

  name: str = ""
  value: Optional[str] = None
  quote_style: str = '"'  # '"', "'", or ""

  def emit(self, indent_level: int = 0) -> str:
    """Emit the attribute.

    Args:
        indent_level: The current indentation level (unused in this node).

    Returns:
        str: The rendered attribute string with leading and trailing trivia.
    """
    if self.value is None:
      return f"{self.leading_trivia}{self.name}{self.trailing_trivia}"
    return f"{self.leading_trivia}{self.name}={self.quote_style}{self.value}{self.quote_style}{self.trailing_trivia}"


@dataclass
class TagNode(HtmlNode):
  """Represents an HTML Element.

  Attributes:
      name: The HTML tag name.
      attributes: A list of attributes for the element.
      children: A list of child nodes contained within the element.
      self_closing: True if the element is self-closing, False otherwise.
  """

  name: str = ""
  attributes: List[AttributeNode] = field(default_factory=list)
  children: List[HtmlNode] = field(default_factory=list)
  self_closing: bool = False

  def append_child(self, child: HtmlNode) -> None:
    """Appends a child node to the end of the children list.

    Args:
        child: The HTML node to append.
    """
    self.children.append(child)

  def remove_child(self, child: HtmlNode) -> None:
    """Removes a child node from the children list.

    Args:
        child: The HTML node to remove.

    Raises:
        ValueError: If the child is not found.
    """
    self.children.remove(child)

  def get_attribute(self, name: str) -> Optional[str]:
    """Gets the value of an attribute by name.

    Args:
        name: The name of the attribute.

    Returns:
        Optional[str]: The value of the attribute, or None if not found or value-less.
    """
    for attr in self.attributes:
      if attr.name == name:
        return attr.value
    return None

  def set_attribute(self, name: str, value: Optional[str] = None) -> None:
    """Sets or updates the value of an attribute by name.

    Args:
        name: The name of the attribute.
        value: The value of the attribute.
    """
    for attr in self.attributes:
      if attr.name == name:
        attr.value = value
        return
    self.attributes.append(AttributeNode(name=name, value=value))

  def remove_attribute(self, name: str) -> None:
    """Removes an attribute by name. Does nothing if not found.

    Args:
        name: The name of the attribute to remove.
    """
    self.attributes = [a for a in self.attributes if a.name != name]

  def emit(self, indent_level: int = 0) -> str:
    """Emit the HTML tag.

    Args:
        indent_level: The indentation level for the tag children.

    Returns:
        str: The fully rendered HTML tag string.
    """
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
  """Represents an SVG connection line between grid cells.

  Attributes:
      x1: The starting X coordinate.
      y1: The starting Y coordinate.
      x2: The ending X coordinate.
      y2: The ending Y coordinate.
      style_class: The CSS class to apply to the line.
      marker_end: The marker-end attribute value for arrowheads.
      parent_style: The CSS style attribute for the enclosing SVG container.
  """

  x1: int = 0
  y1: int = 0
  x2: int = 0
  y2: int = 0
  style_class: str = ""
  marker_end: str = ""
  parent_style: str = ""

  def to_tag(self) -> TagNode:
    """Converts the arrow to a pure CST TagNode.

    Returns:
        TagNode: The corresponding tag element structure representing the SVG.
    """
    line = TagNode(
      name="line",
      self_closing=True,
      attributes=[
        AttributeNode(name="x1", value=str(self.x1)),
        AttributeNode(name="y1", value=str(self.y1)),
        AttributeNode(name="x2", value=str(self.x2)),
        AttributeNode(name="y2", value=str(self.y2)),
        AttributeNode(name="class", value=self.style_class),
        AttributeNode(name="marker-end", value=self.marker_end),
      ],
    )
    return TagNode(
      name="svg",
      attributes=[
        AttributeNode(name="class", value="sw-arrow"),
        AttributeNode(name="style", value=self.parent_style),
      ],
      children=[line],
    )

  def emit(self, indent_level: int = 0) -> str:
    """Renders the arrow as an absolute SVG element.

    Args:
        indent_level: The indentation level for the rendering.

    Returns:
        str: The rendered SVG string representation of the arrow.
    """
    return self.to_tag().emit(indent_level)


@dataclass
class GridBox(HtmlNode):
  """Represents a content box positioned within the CSS Grid.

  Attributes:
      row: The CSS grid row index.
      col: The CSS grid column index.
      css_class: The CSS class to apply to the element.
      header_text: Header label for the box.
      code_text: Optional code block content.
      body_text: Optional body content text.
      arrows: Connection arrows stemming from this grid box.
      z_index: Optional z-index styling.
  """

  row: int = 0
  col: int = 0
  css_class: str = ""
  header_text: str = ""
  code_text: Optional[str] = None
  body_text: Optional[str] = None
  arrows: List[SvgArrow] = field(default_factory=list)
  z_index: Optional[int] = None

  def to_tag(self) -> TagNode:
    """Converts the grid box to a pure CST TagNode.

    Returns:
        TagNode: The corresponding tag element structure for the GridBox.
    """
    style = f"grid-row:{self.row}; grid-column:{self.col};"
    if self.z_index is not None:
      style += f" z-index:{self.z_index};"

    safe_header = escape_html(self.header_text)

    children: List[HtmlNode] = []

    if "circ" in self.css_class:
      children.append(TextNode(content=safe_header))
    else:
      children.append(
        TagNode(
          name="span",
          attributes=[AttributeNode(name="class", value="header-txt")],
          children=[TextNode(content=safe_header)],
        )
      )

    if self.code_text:
      safe_code = escape_html(self.code_text)
      children.append(TagNode(name="code", children=[TextNode(content=safe_code)]))

    if self.body_text:
      safe_body = escape_html(self.body_text.strip())
      children.append(TextNode(content=safe_body))

    for arrow in self.arrows:
      children.append(arrow.to_tag())

    return TagNode(
      name="div",
      attributes=[
        AttributeNode(name="class", value=self.css_class),
        AttributeNode(name="style", value=style),
      ],
      children=children,
    )

  def emit(self, indent_level: int = 0) -> str:
    """Renders the grid cell div, its content, and attached arrows.

    Args:
        indent_level: The indentation level for the children rendering.

    Returns:
        str: The rendered HTML string representation of the grid box.
    """
    return self.to_tag().emit(indent_level)


@dataclass
class HtmlDocument(HtmlNode):
  """Root container for the generated HTML.

  Attributes:
      model_name: Name of the model representing the content.
      children: Child elements inside the HTML document.
  """

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
    """Renders the complete HTML document.

    Args:
        indent_level: The indentation level for the rendering.

    Returns:
        str: The fully rendered HTML document string.
    """
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
