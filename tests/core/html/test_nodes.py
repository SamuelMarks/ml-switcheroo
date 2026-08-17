"""Unit tests for the HTML node classes used in HTML representation generation.

This module contains test cases to verify the correct behavior of basic and complex HTML node
elements, layout boxes, and structural documents used in the ml_switcheroo HTML-based reporting.
"""

import pytest
from ml_switcheroo.core.html.nodes import (
  HtmlNode,
  TextNode,
  CommentNode,
  AttributeNode,
  TagNode,
  SvgArrow,
  GridBox,
  HtmlDocument,
)


def test_htmlnode_emit_not_implemented() -> None:
  """Verifies that the base HtmlNode raises NotImplementedError when emit is called.

  This test ensures that HtmlNode acts as an abstract or base class and cannot be
  emitted directly without a concrete implementation.

  Returns:
      None
  """
  node = HtmlNode()
  with pytest.raises(NotImplementedError):
    node.emit()


def test_textnode() -> None:
  """Verifies that TextNode properly preserves and formats raw content and trivia.

  This test checks that a TextNode constructed with content and leading or trailing
  trivia characters emits the exact concatenation of leading trivia, content, and
  trailing trivia for both emit() and to_html() methods.

  Returns:
      None
  """
  node = TextNode(content="Hello", leading_trivia=" ", trailing_trivia=" ")
  assert node.emit() == " Hello "
  assert node.to_html() == " Hello "


def test_commentnode() -> None:
  """Verifies that CommentNode converts contents into a well-formed HTML comment.

  This test checks that CommentNode wraps its content with standard HTML comment
  syntaxes (<!-- and -->) and properly wraps this structure with any specified
  leading and trailing trivia.

  Returns:
      None
  """
  node = CommentNode(content=" comment ", leading_trivia=" ", trailing_trivia=" ")
  assert node.emit() == " <!-- comment --> "


def test_attributenode() -> None:
  """Verifies that AttributeNode correctly formats key-value and boolean attributes.

  This test ensures that valued HTML attributes are emitted in key="value" format
  preceded by any specified leading trivia, and that boolean/valueless HTML attributes
  are emitted simply as the attribute name.

  Returns:
      None
  """
  node1 = AttributeNode(name="class", value="test", leading_trivia=" ")
  assert node1.emit() == ' class="test"'

  node2 = AttributeNode(name="disabled", value=None)
  assert node2.emit() == "disabled"


def test_tagnode() -> None:
  """Verifies TagNode's attribute and child node management, and emission format.

  This test thoroughly checks that TagNode correctly adds, updates, retrieves, and
  removes HTML attributes, and manages children nodes correctly by appending and
  removing child instances, emitting well-formatted HTML blocks.

  Returns:
      None
  """
  node = TagNode(name="div")
  assert node.emit() == "<div></div>"

  node.set_attribute("class", "box")
  assert node.emit() == '<div class="box"></div>'

  node.set_attribute("class", "container")
  assert node.emit() == '<div class="container"></div>'

  assert node.get_attribute("class") == "container"
  assert node.get_attribute("id") is None

  node.remove_attribute("class")
  assert node.emit() == "<div></div>"

  child = TextNode(content="content")
  node.append_child(child)
  assert node.emit() == "<div>content</div>"

  node.remove_child(child)
  assert node.emit() == "<div></div>"


def test_tagnode_self_closing() -> None:
  """Verifies that self-closing tags are emitted with correct XML-style trailing slashes.

  This test checks that TagNode instances with self_closing=True format their tags as
  single tags ending with "/>" rather than distinct opening and closing tags.

  Returns:
      None
  """
  node = TagNode(name="img", self_closing=True)
  node.set_attribute("src", "test.png")
  assert node.emit() == '<img src="test.png"/>'


def test_svgarrow() -> None:
  """Verifies that SvgArrow renders valid SVG markup representing a directed line.

  This test validates that SvgArrow correctly positions endpoints, applies CSS
  classes, sets appropriate arrow markers, and wraps everything in a root <svg>
  container element.

  Returns:
      None
  """
  arrow = SvgArrow(x1=0, y1=0, x2=10, y2=10, style_class="s-red", marker_end="url(#mr)")
  html = arrow.emit()
  assert "<svg" in html
  assert '<line x1="0" y1="0" x2="10" y2="10" class="s-red" marker-end="url(#mr)"/>' in html


def test_gridbox() -> None:
  """Verifies that GridBox correctly generates structural grids with custom styles and text.

  This test checks that GridBox instances compile CSS grid positioning (rows, columns),
  custom CSS class names, z-index attributes, header labels, inline code segments, and
  inner body text into an HTML division representation.

  Returns:
      None
  """
  box = GridBox(row=1, col=2, css_class="box r", header_text="Test", code_text="x = 1", body_text="body", z_index=5)
  html = box.emit()
  assert '<div class="box r" style="grid-row:1; grid-column:2; z-index:5;">' in html
  assert '<span class="header-txt">Test</span>' in html
  assert "<code>x = 1</code>" in html
  assert "body" in html


def test_gridbox_circ() -> None:
  """Verifies that GridBox can render simplified circle layouts when using circ class.

  This test ensures that when GridBox is initialized with a CSS class matching 'circ',
  it constructs a simplified circular layout with the header text embedded directly
  in the container rather than in a separate header block.

  Returns:
      None
  """
  box = GridBox(row=1, col=2, css_class="circ", header_text="Test")
  html = box.emit()
  assert '<div class="circ" style="grid-row:1; grid-column:2;">Test</div>' in html


def test_htmldocument() -> None:
  """Verifies that HtmlDocument compiles a full HTML boilerplate containing model metadata.

  This test checks that HtmlDocument formats a complete HTML document with proper doctype,
  and structures the visual container with specified model information.

  Returns:
      None
  """
  doc = HtmlDocument(model_name="TestModel")
  html = doc.emit()
  assert "<!DOCTYPE html>" in html
  assert "<h3>Model: TestModel</h3>" in html


def test_htmldocument_pure_cst() -> None:
  """Verifies that HtmlDocument acts as a pure CST transmitter when direct children are given.

  This test ensures that when HtmlDocument is instantiated with specific child nodes and no
  model metadata, it emits the direct inner HTML/CST content instead of generating the
  standard full HTML document layout.

  Returns:
      None
  """
  doc = HtmlDocument(children=[TextNode(content="pure html")])
  html = doc.emit()
  assert html == "pure html"


def test_tagnode_set_attribute_multiple():
  """Verifies set_attribute behavior when other attributes exist."""
  from ml_switcheroo.core.html.nodes import TagNode, AttributeNode

  tag = TagNode(name="div")
  tag.attributes.append(AttributeNode(name="id", value="test"))
  # Loop condition attr.name == name is False
  tag.set_attribute("class", "container")
  # Loop condition attr.name == name is True
  tag.set_attribute("id", "updated")

  assert len(tag.attributes) == 2
  assert tag.get_attribute("id") == "updated"
  assert tag.get_attribute("class") == "container"
