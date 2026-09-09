"""Test suite for the Html Nodes module."""

from ml_switcheroo.core.html.nodes import (
  AttributeNode,
  CommentNode,
  GridBox,
  HtmlDocument,
  HtmlNode,
  SvgArrow,
  TagNode,
  TextNode,
)


def test_svg_arrow_render() -> None:
  """Verifies the behavior of svg arrow render."""
  arrow = SvgArrow(x1=0, y1=0, x2=50, y2=50, style_class="s-red", marker_end="url(#mr)", parent_style="left:100%")
  html: str = arrow.to_html()
  assert '<svg class="sw-arrow" style="left:100%">' in html
  assert '<line x1="0"' in html
  assert 'class="s-red"' in html
  assert 'marker-end="url(#mr)"' in html


def test_grid_box_render_standard() -> None:
  """Verifies the behavior of grid box render standard."""
  arrow = SvgArrow(x1=0, y1=0, x2=10, y2=10, style_class="s-blue", marker_end="", parent_style="")
  box = GridBox(row=2, col=1, css_class="box r", header_text="Header", code_text="x=1", body_text="Body", arrows=[arrow])
  html: str = box.to_html()
  assert 'class="box r"' in html
  assert 'style="grid-row:2; grid-column:1;"' in html
  assert '<span class="header-txt">Header</span>' in html
  assert "<code>x=1</code>" in html
  assert "Body" in html
  assert "<svg" in html


def test_grid_box_render_circle() -> None:
  """Verifies the behavior of grid box render circle."""
  box = GridBox(row=5, col=2, css_class="circ", header_text="Return")
  html: str = box.to_html()
  assert 'class="circ"' in html
  assert "Return" in html
  assert "header-txt" not in html


def test_document_render_structure() -> None:
  """Verifies the behavior of document render structure."""
  b1 = GridBox(row=2, col=1, css_class="b", header_text="A")
  b2 = GridBox(row=3, col=1, css_class="b", header_text="B")
  doc = HtmlDocument(model_name="TestNet", children=[b1, b2])
  html: str = doc.to_html()
  assert "Model: TestNet" in html
  assert "repeat(2, 80px)" in html
  assert ".s-green {" in html
  assert 'marker id="mr"' in html
  assert "Memory (Init)" in html
  assert "sw-grid" in html
  assert ">A</span>" in html
  assert ">B</span>" in html


def test_html_node_base() -> None:
  """Docstring."""
  node = HtmlNode()
  try:
    node.emit()
  except NotImplementedError:
    pass

  try:
    node.to_html()
  except NotImplementedError:
    pass


def test_text_node() -> None:
  """Docstring."""
  node = TextNode(content="hello")
  assert node.emit() == "hello"


def test_comment_node() -> None:
  """Docstring."""
  node = CommentNode(content=" test ")
  assert node.emit() == "<!-- test -->"


def test_attribute_node() -> None:
  """Docstring."""
  attr = AttributeNode(name="class", value="test", quote_style="'")
  assert attr.emit() == "class='test'"
  attr2 = AttributeNode(name="disabled")
  assert attr2.emit() == "disabled"


def test_tag_node() -> None:
  """Docstring."""
  tag = TagNode(name="br", self_closing=True)
  assert tag.emit() == "<br/>"
  tag2 = TagNode(name="div", children=[TextNode(content="hello")])
  assert tag2.emit() == "<div>hello</div>"


def test_tag_node_with_children() -> None:
  """Docstring."""
  child = TagNode(name="span", children=[TextNode(content="A")])
  tag = TagNode(name="div", attributes=[AttributeNode(name="id", value="main")], children=[child])
  html: str = tag.emit()
  assert 'div id="main"' in html
  assert "<span>A</span>" in html


def test_tag_node_manipulation() -> None:
  """Docstring."""
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


def test_html_nodes_remaining_branches() -> None:
  """Test remaining branches in html nodes for 100% branch and line coverage."""
  # 1. get_attribute non-existent and set_attribute non-first match
  tag = TagNode(name="div", attributes=[AttributeNode(name="class", value="foo"), AttributeNode(name="id", value="bar")])
  assert tag.get_attribute("non_existent") is None
  tag.set_attribute("id", "baz")
  assert tag.get_attribute("id") == "baz"

  # 2. attr with leading_trivia
  attr_trivia = AttributeNode(name="id", value="bar", leading_trivia=" ")
  tag_trivia = TagNode(name="span", attributes=[attr_trivia])
  assert tag_trivia.emit() == '<span id="bar"></span>'

  # 3. GridBox with z_index
  box = GridBox(row=1, col=1, z_index=10, header_text="ZBox")
  html_box = box.to_html()
  assert "z-index:10;" in html_box

  # 4. Pure CST HtmlDocument
  pure_doc = HtmlDocument(
    leading_trivia="<!-- header -->",
    trailing_trivia="<!-- footer -->",
    children=[TagNode(name="div", children=[TextNode(content="CST Content")])],
  )
  pure_html = pure_doc.emit()
  assert "<!-- header --><div>CST Content</div><!-- footer -->" == pure_html

  # 5. Empty HtmlDocument
  empty_doc = HtmlDocument(model_name="EmptyModel", children=[])
  empty_html = empty_doc.emit()
  assert "Model: EmptyModel" in empty_html
  assert "repeat(0, 80px)" in empty_html
