"""Unit tests for the HTML-to-DSL parser and grid extractor components.

This module validates the correctness of the HTML parsing logic, the custom
element parsing, the tree structures representing HTML nodes, and the subsequent
extraction of DSL module layers and execution graphs. Specifically, it tests:
1. `InternalHtmlParser` for basic and robust unclosed tag handling.
2. `GridExtractor` for extracting model names, attribute layers, and operations.
3. `HtmlParser` integration for complete parsing to AST and output generation.
"""

from ml_switcheroo.core.html.parser import HtmlParser, InternalHtmlParser, GridExtractor
from ml_switcheroo.core.html.nodes import HtmlDocument, TagNode, TextNode, AttributeNode


def test_internal_html_parser_basic():
  """Tests basic tag, attribute, text, and self-closing node parsing.

  Feeds a well-formed HTML segment containing nested div, image, and line-break
  elements, comments, and a DOCTYPE tag into the `InternalHtmlParser`, then
  asserts that the resulting parser node tree matches the expected structure.

  Args:
      None

  Returns:
      None
  """
  parser = InternalHtmlParser()
  parser.feed('<div class="test">Hello<img src="img.png"/><br></div><!-- comment --><!DOCTYPE html>')

  assert len(parser.root_children) == 3
  div = parser.root_children[0]
  assert isinstance(div, TagNode)
  assert div.name == "div"
  assert len(div.attributes) == 1
  assert div.attributes[0].name == "class"
  assert div.attributes[0].value == "test"

  assert len(div.children) == 3
  assert isinstance(div.children[0], TextNode)
  assert div.children[0].content == "Hello"

  img = div.children[1]
  assert isinstance(img, TagNode)
  assert img.name == "img"
  assert img.self_closing is True

  br = div.children[2]
  assert isinstance(br, TagNode)
  assert br.name == "br"
  assert br.self_closing is True


def test_internal_html_parser_unclosed():
  """Tests parsing behaviour with unclosed nested elements.

  Feeds HTML where an inner `span` tag is never closed before the outer
  `div` tag is closed. Verifies that the parser correctly reconstructs
  the logical tree by nested nesting.

  Args:
      None

  Returns:
      None
  """
  parser = InternalHtmlParser()
  parser.feed("<div><span>Text</div>")

  assert len(parser.root_children) == 1
  div = parser.root_children[0]
  assert len(div.children) == 1
  span = div.children[0]
  assert span.name == "span"
  assert len(span.children) == 1
  assert span.children[0].content == "Text"


def test_internal_html_parser_unclosed_root():
  """Tests closing out dangling tags at the root of the document.

  Feeds HTML with an unclosed outer `div` element, then manually pops
  and closes dangling tags from the stack (mirroring how `HtmlParser`
  finalizes incomplete input) to verify that nodes are nested correctly.

  Args:
      None

  Returns:
      None
  """
  parser = InternalHtmlParser()
  parser.feed("<div>Text")

  # Needs to be closed out manually at end like HtmlParser does
  while parser.stack:
    unclosed = parser.stack.pop()
    if parser.stack:
      parser.stack[-1].children.append(unclosed)
    else:
      parser.root_children.append(unclosed)

  assert len(parser.root_children) == 1
  div = parser.root_children[0]
  assert div.name == "div"
  assert len(div.children) == 1
  assert div.children[0].content == "Text"


def test_grid_extractor_basic():
  """Tests extracting model layers, stateful calls, and functional calls.

  Constructs a complete manual `HtmlDocument` containing model attribute metadata
  and operation calls, and executes the `GridExtractor` to verify the parsed
  attributes (e.g. `conv: Conv2d`) and sequential operations (e.g. `Call (conv)`, `Relu`)
  are extracted correctly.

  Args:
      None

  Returns:
      None
  """
  doc = HtmlDocument(model_name="MyModel")

  # Add an attribute (layer)
  attr_box = TagNode(name="div", attributes=[AttributeNode(name="class", value="box r")])
  attr_box.append_child(TagNode(name="span", children=[TextNode(content="conv: Conv2d")]))
  attr_box.append_child(TagNode(name="code", children=[TextNode(content="args: x")]))

  # Add a stateful call
  op_box = TagNode(name="div", attributes=[AttributeNode(name="class", value="box b")])
  op_box.append_child(TagNode(name="span", children=[TextNode(content="Call (conv)")]))
  op_box.append_child(TagNode(name="code", children=[TextNode(content="args: x")]))

  # Add a functional call
  func_box = TagNode(name="div", attributes=[AttributeNode(name="class", value="box b")])
  func_box.append_child(TagNode(name="span", children=[TextNode(content="Relu")]))
  func_box.append_child(TagNode(name="code", children=[TextNode(content="")]))

  doc.children = [attr_box, op_box, func_box]

  extractor = GridExtractor()
  extractor.extract(doc)

  assert extractor.model_name == "MyModel"
  assert len(extractor.attrs) == 1
  assert extractor.attrs[0] == ("conv", "Conv2d", "args: x")

  assert len(extractor.ops) == 2
  assert extractor.ops[0] == ("Call (conv)", "args: x")
  assert extractor.ops[1] == ("Relu", "")


def test_grid_extractor_header_parsing():
  """Tests extraction of the model name from h3 headers.

  Validates that when the HTML contains a header element of form
  `<h3>Model: CustomNet</h3>`, the `GridExtractor` correctly detects and parses
  the name "CustomNet" as the target model name.

  Args:
      None

  Returns:
      None
  """
  doc = HtmlDocument()
  h3 = TagNode(name="h3", children=[TextNode(content="Model: CustomNet")])
  doc.children = [h3]

  extractor = GridExtractor()
  extractor.extract(doc)

  assert extractor.model_name == "CustomNet"


def test_grid_extractor_attr_no_colon():
  """Tests extraction of attributes missing the class name colon separator.

  Validates that if an attribute box does not contain a colon in its span tag
  (e.g., `just_name` instead of `just_name: Linear`), the `GridExtractor`
  defaults the attribute type to "Unknown" and continues gracefully.

  Args:
      None

  Returns:
      None
  """
  doc = HtmlDocument()
  attr_box = TagNode(name="div", attributes=[AttributeNode(name="class", value="box r")])
  attr_box.append_child(TagNode(name="span", children=[TextNode(content="just_name")]))
  attr_box.append_child(TagNode(name="code", children=[TextNode(content="")]))
  doc.children = [attr_box]

  extractor = GridExtractor()
  extractor.extract(doc)

  assert len(extractor.attrs) == 1
  assert extractor.attrs[0] == ("just_name", "Unknown", "")


def test_htmlparser_integration():
  """Tests the full compilation pipeline from HTML string to python DSL module.

  Feeds a multi-line HTML string representing a neural network layer structure and
  execution pipeline to `HtmlParser`, and asserts that the compiled CST module code
  contains correct module initialization, layer declarations, forward pass definitions,
  and sequential execution flow matching the input.

  Args:
      None

  Returns:
      None
  """
  html = """
    <h3>Model: TestModel</h3>
    <div class="box r"><span>conv: Conv2d</span><code>args: x</code></div>
    <div class="box b"><span>Call (conv)</span><code>args: x</code></div>
    <div class="box b"><span>Relu</span><code></code></div>
    <div class="box b"><span>Add</span><code>x, y=2</code></div>
    """

  parser = HtmlParser(html)
  cst_mod = parser.parse()

  code = cst_mod.code

  assert "class TestModel(dsl.Module):" in code
  assert "self.conv = dsl.Conv2d()" in code
  assert "conv_out = self.conv(x)" in code
  assert "relu_out = dsl.Relu(conv_out)" in code
  assert "add_out = dsl.Add(relu_out, x, y=2)" in code
  assert "return add_out" in code


def test_htmlparser_empty_config():
  """Tests that declaring layers with empty code/argument sections works.

  Validates that layers specified with empty arguments lists are declared
  and instantiated without trailing commas or syntax errors in Python (e.g.
  `self.layer1 = dsl.Linear()`).

  Args:
      None

  Returns:
      None
  """
  html = """
    <div class="box r"><span>layer1: Linear</span><code></code></div>
    """
  parser = HtmlParser(html)
  cst_mod = parser.parse()
  code = cst_mod.code
  assert "self.layer1 = dsl.Linear()" in code


def test_htmlparser_safe_val_fallback():
  """Tests parser fallback mechanisms when meeting unparseable arguments.

  Verifies that if an operation arguments code contains non-Python syntax,
  the `_safe_val` parsing logic catches the error and compiles it safely
  by wrapping the raw argument string in Python string quotes as a fallback.

  Args:
      None

  Returns:
      None
  """
  # Provide an invalid python expression for arguments to trigger _safe_val fallback
  html = """
    <div class="box b"><span>Op</span><code>invalid syntax args</code></div>
    """
  parser = HtmlParser(html)
  cst_mod = parser.parse()
  code = cst_mod.code
  assert "dsl.Op(x, 'invalid syntax args')" in code


def test_htmlparser_empty():
  """Tests compiling an empty HTML document.

  Validates that parsing empty HTML defaults safely to a skeletal model class definition
  named `Model` with a `pass` body and a default `forward` method returning its input.

  Args:
      None

  Returns:
      None
  """
  html = ""
  parser = HtmlParser(html)
  cst_mod = parser.parse()
  code = cst_mod.code
  assert "class Model(dsl.Module):" in code
  assert "pass" in code
  assert "return x" in code


def test_internal_parser_missing_attribute_value():
  """Tests parsing HTML with attributes that have no value."""
  parser = InternalHtmlParser()
  # Use a self-closing tag with a valueless attribute and a normal tag with a valueless attribute
  parser.feed('<div disabled><img src="test.png" defer/></div>')

  assert len(parser.root_children) == 1
  div = parser.root_children[0]
  assert div.attributes[0].name == "disabled"
  assert div.attributes[0].value is None

  img = div.children[0]
  assert img.attributes[1].name == "defer"
  assert img.attributes[1].value is None


def test_htmlparser_safe_val_eval_exception():
  """Tests fallback when eval throws an error but parse_expression succeeds."""
  html = """
    <div class="box b"><span>Op</span><code>some_unknown_var</code></div>
  """
  parser = HtmlParser(html)
  cst_mod = parser.parse()
  code = cst_mod.code
  # "some_unknown_var" parses as Name, but eval("some_unknown_var") raises NameError.
  # So it should return the parsed expression rather than evaluating and converting to literal.
  assert "dsl.Op(x, some_unknown_var)" in code
