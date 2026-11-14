"""Unit tests for HTML CST nodes and `HtmlParser` / `GridExtractor` (`ml_switcheroo.core.html`)."""

import libcst as cst
import pytest

from ml_switcheroo.core.cst.base import Trivia
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
from ml_switcheroo.core.html.parser import GridExtractor, HtmlParser, InternalHtmlParser


def test_html_nodes_emission_and_attributes() -> None:
  """Verify `HtmlNode`, `TextNode`, `CommentNode`, `AttributeNode`, `TagNode`, `SvgArrow`, `GridBox`, and `HtmlDocument`."""
  base = HtmlNode()
  with pytest.raises(NotImplementedError):
    base.emit()
  with pytest.raises(NotImplementedError):
    base.to_html()

  assert TextNode(content="hello", leading_trivia=" ", trailing_trivia=" ").emit() == " hello "
  assert CommentNode(content="comment", leading_trivia="  ").emit() == "  <!--comment-->"
  assert AttributeNode(name="class", value="box", quote_style="'").emit() == "class='box'"
  assert AttributeNode(name="disabled", value=None).emit() == "disabled"

  tag = TagNode(name="div")
  assert tag.emit() == "<div></div>"
  tag.set_attribute("class", "box")
  assert tag.get_attribute("class") == "box"
  tag.set_attribute("class", "box2")
  assert tag.get_attribute("class") == "box2"
  assert tag.get_attribute("missing") is None
  tag.remove_attribute("class")
  assert tag.get_attribute("class") is None
  tag.remove_attribute("missing")

  child = TextNode(content="hi")
  tag.append_child(child)
  assert tag.emit() == "<div>hi</div>"
  tag.remove_child(child)
  assert tag.emit() == "<div></div>"
  with pytest.raises(ValueError, match="not in list"):
    tag.remove_child(child)

  tag.self_closing = True
  tag.set_attribute("class", "x")
  tag.attributes[0].leading_trivia = [Trivia(" ")]
  assert tag.emit() == '<div class="x"/>'
  tag.attributes[0].leading_trivia = []
  assert tag.emit() == '<div class="x"/>'

  arrow = SvgArrow(x1=0, y1=1, x2=2, y2=3, style_class="s-blue", marker_end="url(#m)", parent_style="display:none")
  assert arrow.to_tag().name == "svg"
  assert "s-blue" in arrow.emit()

  box = GridBox(
    row=1, col=2, css_class="box r", header_text="Call (conv)", code_text="args: x", body_text="Hello", z_index=5
  )
  assert box.to_tag().name == "div"
  assert "Call (conv)" in box.emit()

  circ = GridBox(css_class="circ", header_text="X")
  assert "header-txt" not in circ.emit()
  assert "X" in circ.emit()

  doc = HtmlDocument(model_name="TestModel")
  assert "TestModel" in doc.emit()
  assert "<!DOCTYPE html>" in doc.emit()
  doc.children = [GridBox(row=2, css_class="box")]
  assert "grid-template-rows: 30px repeat(1, 80px);" in doc.emit()
  doc.children = [TagNode(name="br", self_closing=True)]
  assert "<br/>" in doc.emit()


def test_internal_html_parser_and_grid_extractor_edge_cases() -> None:
  """Verify `InternalHtmlParser` unclosed/valueless/doctype tags and `GridExtractor` DOM filtering."""
  parser = InternalHtmlParser()
  parser.feed("<div><!--cmt-->Text<br/><img></div>")
  assert len(parser.root_children) == 1
  div = parser.root_children[0]
  assert isinstance(div, TagNode)
  assert len(div.children) == 4

  parser.feed("<!DOCTYPE html>")
  parser.feed("<p>")
  assert len(parser.stack) == 1
  doctype = parser.root_children[1]
  assert isinstance(doctype, TagNode)
  assert doctype.name == "!DOCTYPE html"
  assert doctype.self_closing is True

  p2 = InternalHtmlParser()
  p2.feed('<input disabled class="input-class">')
  in_tag = p2.root_children[0]
  assert isinstance(in_tag, TagNode)
  assert any(a.name == "disabled" and a.value is None for a in in_tag.attributes)
  p2.handle_startendtag("custom-tag", [("valueless", None), ("valued", "123")])
  custom = p2.root_children[1]
  assert isinstance(custom, TagNode)
  assert custom.self_closing is True

  p3 = InternalHtmlParser()
  p3.handle_endtag("nonexistent")
  assert p3.root_children == []
  p3.feed("<section><p><span>text</section>")
  sec = p3.root_children[0]
  assert isinstance(sec, TagNode)
  assert sec.name == "section"
  assert len(sec.children) >= 1

  unclosed_doc = HtmlParser("<div><span>unclosed text").parse_cst()
  outer_div = unclosed_doc.children[0]
  assert isinstance(outer_div, TagNode)
  assert outer_div.name == "div"
  inner_span = outer_div.children[0]
  assert isinstance(inner_span, TagNode)
  assert inner_span.name == "span"
  inner_txt = inner_span.children[0]
  assert isinstance(inner_txt, TextNode)
  assert inner_txt.content == "unclosed text"

  ext = GridExtractor()
  ext.extract(HtmlDocument(model_name="", children=[]))
  assert ext.model_name == "Model"

  rich_doc = HtmlDocument(
    model_name="Test",
    children=[
      TextNode(content="trivia"),
      CommentNode(content="comment"),
      TagNode(name="h3", children=[TagNode(name="b"), TextNode(content="NonModel")]),
      TagNode(name="h3", children=[TextNode(content="Model: AModel")]),
      TagNode(name="div", attributes=[AttributeNode(name="id", value="ignored")]),
      TagNode(name="div", attributes=[AttributeNode(name="class", value=None)]),
      TagNode(name="div", attributes=[AttributeNode(name="class", value="container")]),
      TagNode(
        name="div",
        attributes=[AttributeNode(name="class", value="box ignored_color")],
        children=[
          TextNode(content="ws"),
          TagNode(name="i", children=[TextNode(content="icon")]),
          TagNode(name="span", children=[TagNode(name="strong")]),
          TagNode(name="code", children=[TagNode(name="em")]),
        ],
      ),
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
          TagNode(name="span", children=[TextNode(content="Call")]),
          TagNode(name="code", children=[TextNode(content="")]),
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
      TagNode(name="article", children=[TextNode(content="article")]),
    ],
  )
  ext2 = GridExtractor()
  ext2.extract(rich_doc)
  assert ext2.model_name == "AModel"
  assert ext2.attrs == [("myattr", "Conv2d", "kernel_size=3"), ("unknown_attr", "Unknown", "")]
  assert ext2.ops == [("Call (myattr)", "args: x"), ("relu", "alpha=0.1")]


def test_html_parser_facade_and_arg_lowering() -> None:
  """Verify `HtmlParser.parse`, `_parse_args_str`, `_create_call`, and `_safe_val`."""
  p_helper = HtmlParser("")
  assert p_helper._parse_args_str("") == []
  assert len(p_helper._create_call("dsl.Linear", config_str=None).args) == 0
  parsed_args = p_helper._parse_args_str("123, axis=1, True")
  assert len(parsed_args) == 3
  assert parsed_args[1].keyword is not None
  assert parsed_args[1].keyword.value == "axis"
  assert isinstance(p_helper._safe_val("def invalid_syntax(): pass"), cst.SimpleString)

  html = """
  <!-- header comment -->
  <h3>Section Header</h3>
  <h3>Model: TestModel</h3>
  <div class="box r"><span>my_layer: Linear</span><code>in_features=10, out_features=5</code></div>
  <div class="box r"><span>empty_layer: Empty</span><code>args: </code></div>
  <div class="box b"><span>Call (my_layer)</span><code>args: x</code></div>
  <div class="box b"><span>Call (empty_layer)</span><code></code></div>
  <div class="box b"><span>relu</span><code></code></div>
  <div class="box b"><span>tanh</span><code>x, !!error, foo='bar'</code></div>
  <div class="box b"><span>CallWrong</span><code></code></div>
  """
  code = HtmlParser(html).parse().code
  assert "class TestModel" in code
  assert "my_layer = dsl.Linear(in_features=10, out_features=5)" in code
  assert "empty_layer = dsl.Empty()" in code
  assert "my_layer_out = self.my_layer(x)" in code
  assert "relu_out = dsl.relu(empty_layer_out)" in code

  assert "pass" in HtmlParser("<h3>Model: EmptyModel</h3>").parse().code
