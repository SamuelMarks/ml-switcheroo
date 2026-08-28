"""Module docstring."""

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
from ml_switcheroo.core.html.parser import InternalHtmlParser, GridExtractor, HtmlParser
import libcst as cst


def test_nodes_base_html_node() -> None:
  """Docstring."""
  node: HtmlNode = HtmlNode()
  with pytest.raises(NotImplementedError):
    node.emit()
  with pytest.raises(NotImplementedError):
    node.to_html()


def test_nodes_text_node() -> None:
  """Docstring."""
  node: TextNode = TextNode(content="hello", leading_trivia=" ", trailing_trivia=" ")
  assert node.emit() == " hello "


def test_nodes_comment_node() -> None:
  """Docstring."""
  node: CommentNode = CommentNode(content="comment", leading_trivia="  ")
  assert node.emit() == "  <!--comment-->"


def test_nodes_attribute_node() -> None:
  """Docstring."""
  node: AttributeNode = AttributeNode(name="class", value="box", quote_style="'")
  assert node.emit() == "class='box'"

  node_valueless: AttributeNode = AttributeNode(name="disabled", value=None)
  assert node_valueless.emit() == "disabled"


def test_nodes_tag_node() -> None:
  """Docstring."""
  tag: TagNode = TagNode(name="div")
  assert tag.emit() == "<div></div>"

  tag.set_attribute("class", "box")
  assert tag.get_attribute("class") == "box"

  tag.set_attribute("class", "box2")
  assert tag.get_attribute("class") == "box2"

  assert tag.get_attribute("missing") is None

  tag.remove_attribute("class")
  assert tag.get_attribute("class") is None

  tag.remove_attribute("missing")

  child: TextNode = TextNode(content="hi")
  tag.append_child(child)
  assert tag.emit() == "<div>hi</div>"

  tag.remove_child(child)
  assert tag.emit() == "<div></div>"

  with pytest.raises(ValueError):
    tag.remove_child(child)

  tag.self_closing = True
  tag.set_attribute("class", "x")
  # need coverage for self-closing without trailing trivia and with leading trivia in attr
  tag.attributes[0].leading_trivia = " "
  assert tag.emit() == '<div class="x"/>'
  tag.attributes[0].leading_trivia = ""
  assert tag.emit() == '<div class="x"/>'


def test_nodes_svg_arrow() -> None:
  """Docstring."""
  arrow: SvgArrow = SvgArrow(
    x1=0, y1=1, x2=2, y2=3, style_class="s-blue", marker_end="url(#m)", parent_style="display:none"
  )
  tag: TagNode = arrow.to_tag()
  assert tag.name == "svg"
  assert "s-blue" in arrow.emit()


def test_nodes_grid_box() -> None:
  """Docstring."""
  box: GridBox = GridBox(
    row=1, col=2, css_class="box r", header_text="Call (conv)", code_text="args: x", body_text="Hello", z_index=5
  )
  tag: TagNode = box.to_tag()
  assert tag.name == "div"
  assert "Call (conv)" in box.emit()

  box2: GridBox = GridBox(css_class="circ", header_text="X")
  assert "header-txt" not in box2.emit()
  assert "X" in box2.emit()


def test_nodes_html_document() -> None:
  """Docstring."""
  doc: HtmlDocument = HtmlDocument(model_name="TestModel")
  html: str = doc.emit()
  assert "TestModel" in html
  assert "<!DOCTYPE html>" in html

  doc.children = [GridBox(row=2, css_class="box")]
  html = doc.emit()
  assert "grid-template-rows: 30px repeat(1, 80px);" in html

  doc.children = [TagNode(name="br", self_closing=True)]
  assert "<br/>" in doc.emit()


def test_parser_internal() -> None:
  """Docstring."""
  parser: InternalHtmlParser = InternalHtmlParser()
  parser.feed("<div><!--cmt-->Text<br/><img></div>")
  assert len(parser.root_children) == 1
  div: HtmlNode = parser.root_children[0]
  assert isinstance(div, TagNode)
  assert len(div.children) == 4
  assert isinstance(div.children[0], CommentNode)
  assert isinstance(div.children[1], TextNode)
  assert isinstance(div.children[2], TagNode)

  parser.feed("<!DOCTYPE html>")
  parser.feed("<p>")  # unclosed
  assert len(parser.stack) == 1
  assert len(parser.root_children) == 2
  assert isinstance(parser.root_children[1], TagNode)
  assert parser.root_children[1].name == "!DOCTYPE html"
  assert parser.root_children[1].self_closing


def test_parser_grid_extractor() -> None:
  """Docstring."""
  doc: HtmlDocument = HtmlDocument(
    model_name="Test",
    children=[
      TagNode(name="h3", children=[TextNode(content="Model: AModel")]),
      TagNode(
        name="div",
        attributes=[AttributeNode(name="class", value="box r")],
        children=[
          TagNode(name="span", children=[TextNode(content="myattr: Conv2d")]),
          TagNode(name="code", children=[TextNode(content="kernel_size=3")]),
        ],
      ),
      TagNode(
        name="div",
        attributes=[AttributeNode(name="class", value="box r")],
        children=[
          TagNode(name="span", children=[TextNode(content="unknown_attr")]),
          TagNode(name="code", children=[TextNode(content="")]),
        ],
      ),
      TagNode(
        name="div",
        attributes=[AttributeNode(name="class", value="box b")],
        children=[
          TagNode(name="span", children=[TextNode(content="Call (myattr)")]),
          TagNode(name="code", children=[TextNode(content="args: x")]),
        ],
      ),
      TagNode(
        name="div",
        attributes=[AttributeNode(name="class", value="box b")],
        children=[
          TagNode(name="span", children=[TextNode(content="relu")]),
          TagNode(name="code", children=[TextNode(content="alpha=0.1")]),
        ],
      ),
    ],
  )
  ext: GridExtractor = GridExtractor()
  ext.extract(doc)
  assert ext.model_name == "AModel"
  assert len(ext.attrs) == 2
  assert ext.attrs[0] == ("myattr", "Conv2d", "kernel_size=3")
  assert ext.attrs[1] == ("unknown_attr", "Unknown", "")
  assert len(ext.ops) == 2
  assert ext.ops[0] == ("Call (myattr)", "args: x")
  assert ext.ops[1] == ("relu", "alpha=0.1")


def test_parser_facade() -> None:
  """Docstring."""
  html: str = """
    <h3>Model: TestModel</h3>
    <div class="box r"><span>my_layer: Linear</span><code>in_features=10, out_features=5</code></div>
    <div class="box r"><span>empty_layer: Empty</span><code>args: </code></div>
    <div class="box b"><span>Call (my_layer)</span><code>args: x</code></div>
    <div class="box b"><span>Call (empty_layer)</span><code></code></div>
    <div class="box b"><span>relu</span><code>x</code></div>
    <div class="box b"><span>tanh</span><code>x, !!error, foo='bar'</code></div>
    <div class="box b"><span>CallWrong</span><code></code></div>
    """
  parser: HtmlParser = HtmlParser(html)
  tree: cst.Module = parser.parse()
  assert isinstance(tree, cst.Module)
  code: str = tree.code
  assert "class TestModel" in code
  assert "my_layer = dsl.Linear(in_features=10, out_features=5)" in code
  assert "empty_layer = dsl.Empty()" in code
  assert "my_layer_out = self.my_layer(x)" in code
  assert "relu_out = dsl.relu(empty_layer_out)" in code


def test_parser_facade_no_init() -> None:
  """Docstring."""
  html: str = """<h3>Model: TestModel</h3>"""
  parser: HtmlParser = HtmlParser(html)
  tree: cst.Module = parser.parse()
  code: str = tree.code
  assert "pass" in code


def test_parser_unclosed_tags() -> None:
  """Docstring."""
  parser: HtmlParser = HtmlParser("<div><span>unclosed text")
  doc: HtmlDocument = parser.parse_cst()
  assert len(doc.children) == 1
  assert getattr(doc.children[0], "name", None) == "div"
  assert getattr(getattr(doc.children[0], "children", [None])[0], "name", None) == "span"
  assert (
    getattr(getattr(getattr(doc.children[0], "children", [None])[0], "children", [None])[0], "content", None)
    == "unclosed text"
  )


def test_internal_parser_endtag() -> None:
  """Docstring."""
  parser: InternalHtmlParser = InternalHtmlParser()
  parser.feed("<div><span>text</span></div>")
  assert len(parser.root_children) == 1
  assert getattr(parser.root_children[0], "name", None) == "div"


def test_parser_facade_ops_missing_args() -> None:
  """Docstring."""
  html: str = """
    <h3>Model: TestModel</h3>
    <div class="box b"><span>relu</span><code></code></div>
    """
  parser: HtmlParser = HtmlParser(html)
  tree: cst.Module = parser.parse()
  assert "relu_out = dsl.relu(x)" in tree.code
