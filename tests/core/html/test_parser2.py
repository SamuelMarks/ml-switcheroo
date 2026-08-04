"""Unit tests for the HTML parsing components within ml_switcheroo.

This module verifies correct parsing behaviors for:
- Void HTML elements (such as `<br>`, `<img>`, and `<input>`).
- HTML markup declarations (such as `<!DOCTYPE html>`).
- HTML comment nodes (such as `<!-- comment -->`).
- Arguments string parsing on the high-level `HtmlParser`.
- Creation of syntax tree call structures under empty conditions.
"""

from ml_switcheroo.core.html.parser import InternalHtmlParser, HtmlParser


def test_internal_html_parser_void_elements():
  """Verifies that the internal HTML parser correctly identifies void elements.

  This test feeds a string of self-closing void elements (<br>, <img>, and
  <input>) into the parser and asserts that they are successfully treated as
  self-closing tag nodes and appended to the parsed root tree level.

  Args:
    None

  Returns:
    None
  """
  parser = InternalHtmlParser()
  parser.feed('<br><img src="test.png"><input>')

  assert len(parser.root_children) == 3
  br = parser.root_children[0]
  assert br.name == "br"
  assert br.self_closing is True

  img = parser.root_children[1]
  assert img.name == "img"
  assert img.self_closing is True

  input_tag = parser.root_children[2]
  assert input_tag.name == "input"
  assert input_tag.self_closing is True


def test_internal_html_parser_decl():
  """Verifies that the internal HTML parser parses markup declarations.

  This test feeds an HTML declaration tag (<!DOCTYPE html>) and validates that
  the parser registers the declaration node, preserves its exact name structure,
  and automatically marks the node as self-closing.

  Args:
    None

  Returns:
    None
  """
  parser = InternalHtmlParser()
  parser.feed("<!DOCTYPE html>")
  assert len(parser.root_children) == 1
  decl = parser.root_children[0]
  assert decl.name == "!DOCTYPE html"
  assert decl.self_closing is True


def test_internal_html_parser_comment():
  """Verifies that the internal HTML parser correctly processes comments.

  This test feeds an HTML comment and confirms that the parser creates a
  comment node with the inner text correctly stripped of the comment boundary
  delimiters while preserving leading/trailing whitespace inside.

  Args:
    None

  Returns:
    None
  """
  parser = InternalHtmlParser()
  parser.feed("<!-- This is a comment -->")
  assert len(parser.root_children) == 1
  assert parser.root_children[0].content == " This is a comment "


def test_htmlparser_parse_args_str_empty():
  """Verifies that parsing an empty argument string results in an empty list.

  This test exercises the internal argument parsing logic of `HtmlParser` with
  an empty input string and asserts that it gracefully returns an empty list
  without producing syntax or value errors.

  Args:
    None

  Returns:
    None
  """
  parser = HtmlParser("")
  args = parser._parse_args_str("")
  assert args == []


def test_htmlparser_create_call_empty():
  """Verifies that creating a function call with empty arguments is successful.

  This test validates that `HtmlParser._create_call` correctly constructs a API
  call structure for a specified function path (e.g. "my.func"), splitting it
  properly into its identifier components while ensuring its argument list
  remains empty.

  Args:
    None

  Returns:
    None
  """
  parser = HtmlParser("")
  call = parser._create_call("my.func")
  assert call.func.value.value == "my"
  assert call.func.attr.value == "func"
  assert call.args == []
