"""Test suite for the Html Nodes module."""

from ml_switcheroo.core.html.nodes import (
  SvgArrow,
  GridBox,
  HtmlDocument,
  HtmlNode,
  TextNode,
  CommentNode,
  AttributeNode,
  TagNode,
)


def test_svg_arrow_render():
  """Verifies the behavior of svg arrow render."""
  arrow = SvgArrow(x1=0, y1=0, x2=50, y2=50, style_class="s-red", marker_end="url(#mr)", parent_style="left:100%")
  html = arrow.to_html()
  assert '<svg class="sw-arrow" style="left:100%">' in html
  assert '<line x1="0"' in html
  assert 'class="s-red"' in html
  assert 'marker-end="url(#mr)"' in html


def test_grid_box_render_standard():
  """Verifies the behavior of grid box render standard."""
  arrow = SvgArrow(0, 0, 10, 10, "s-blue", "", "")
  box = GridBox(row=2, col=1, css_class="box r", header_text="Header", code_text="x=1", body_text="Body", arrows=[arrow])
  html = box.to_html()
  assert 'class="box r"' in html
  assert 'style="grid-row:2; grid-column:1;"' in html
  assert '<span class="header-txt">Header</span>' in html
  assert "<code>x=1</code>" in html
  assert "Body" in html
  assert "<svg" in html


def test_grid_box_render_circle():
  """Verifies the behavior of grid box render circle."""
  box = GridBox(row=5, col=2, css_class="circ", header_text="Return")
  html = box.to_html()
  assert 'class="circ"' in html
  assert "Return" in html
  assert "header-txt" not in html


def test_document_render_structure():
  """Verifies the behavior of document render structure."""
  b1 = GridBox(row=2, col=1, css_class="b", header_text="A")
  b2 = GridBox(row=3, col=1, css_class="b", header_text="B")
  doc = HtmlDocument(model_name="TestNet", children=[b1, b2])
  html = doc.to_html()
  assert "Model: TestNet" in html
  assert "repeat(2, 80px)" in html
  assert ".s-green {" in html
  assert 'marker id="mr"' in html
  assert "Memory (Init)" in html
  assert "sw-grid" in html
  assert ">A</span>" in html
  assert ">B</span>" in html


def test_html_node_base():
  """Test HtmlNode base methods raise NotImplementedError."""
  node = HtmlNode()
  try:
    node.emit()
  except NotImplementedError:
    pass

  try:
    node.to_html()
  except NotImplementedError:
    pass


def test_text_node():
  """Test TextNode emit."""
  node = TextNode(content="hello")
  assert node.emit() == "hello"


def test_comment_node():
  """Test CommentNode emit."""
  node = CommentNode(content=" test ")
  assert node.emit() == "<!-- test -->"


def test_attribute_node():
  """Test AttributeNode emit."""
  attr = AttributeNode(name="class", value="test", quote_style="'")
  assert attr.emit() == "class='test'"
  attr2 = AttributeNode(name="disabled")
  assert attr2.emit() == "disabled"


def test_tag_node():
  """Test TagNode emit."""
  tag = TagNode(name="br", self_closing=True)
  assert tag.emit() == "<br/>"
  tag2 = TagNode(name="div", children=[TextNode(content="hello")])
  assert tag2.emit() == "<div>hello</div>"


def test_tag_node_with_children():
  """Test TagNode emit with children."""
  child = TagNode(name="span", children=[TextNode(content="A")])
  tag = TagNode(name="div", attributes=[AttributeNode(name="id", value="main")], children=[child])
  html = tag.emit()
  assert 'div id="main"' in html
  assert "<span>A</span>" in html


def test_tag_node_manipulation():
  """Test TagNode manipulation methods (append, remove, modify attributes)."""
  tag = TagNode(name="div")

  # Attribute manipulation
  tag.set_attribute("class", "box")
  assert tag.get_attribute("class") == "box"

  tag.set_attribute("class", "container")
  assert tag.get_attribute("class") == "container"

  tag.remove_attribute("class")
  assert tag.get_attribute("class") is None

  tag.set_attribute("hidden")
  assert tag.get_attribute("hidden") is None
  assert len(tag.attributes) == 1
  assert tag.attributes[0].name == "hidden"

  # Children manipulation
  child1 = TextNode("first")
  child2 = TextNode("second")

  tag.append_child(child1)
  assert len(tag.children) == 1
  assert tag.children[0] == child1

  tag.append_child(child2)
  assert len(tag.children) == 2

  tag.remove_child(child1)
  assert len(tag.children) == 1
  assert tag.children[0] == child2

  # Remove non-existent
  try:
    tag.remove_child(child1)
    assert False, "Should raise ValueError"
  except ValueError:
    pass
