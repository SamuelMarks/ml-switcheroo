"""Test suite for the Tikz Gap module."""

import pytest
from ml_switcheroo.core.tikz.parser import TikzParser, TokenKind


def test_tikz_end_command():
  """Verifies the behavior of TikZ end command."""
  parser = TikzParser("\\end{tikzpicture}")
  parser.parse()


def test_tikz_peek_eof():
  """Verifies the behavior of TikZ peek eof."""
  parser = TikzParser("\\node (a) {};")
  token = parser._peek(offset=100)
  assert token.kind == TokenKind.EOF


def test_tikz_expect_error():
  """Verifies the behavior of TikZ expect correctly handling an error."""
  parser = TikzParser("\\node (a) {};")
  with pytest.raises(SyntaxError):
    parser._expect(TokenKind.LBRACE)


def test_tikz_node_at_coordinates():
  """Verifies the behavior of TikZ node at coordinates."""
  parser = TikzParser("\\node (a) at (1, 2) {Linear};")
  graph = parser.parse()
  assert len(graph.nodes) == 1


def test_tikz_edge_unexpected_connector():
  """Verifies the behavior of TikZ edge unexpected connector."""
  parser = TikzParser("\\draw (a) (b);")
  graph = parser.parse()
  assert len(graph.edges) == 0


def test_tikz_scan_until_semicolon_eof():
  """Verifies the behavior of TikZ scan until semicolon eof."""
  parser = TikzParser("\\draw (a) ")
  parser._scan_until_semicolon()
  assert parser._is_eof()


def test_tikz_extract_metadata_empty():
  """Verifies the behavior of TikZ extract metadata empty."""
  parser = TikzParser("\\node (a) {\\textbf};")
  graph = parser.parse()
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Unknown"


def test_tikz_parser_gaps():
  """Verifies the behavior of TikZ parser gaps."""
  from ml_switcheroo.core.tikz.parser import TikzParser

  source = "\n\\begin{tikzpicture}\n\\unknowncommand\n\\end{tikzpicture}\n"
  parser = TikzParser(source)
  parser.parse()
  source2 = "\n\\begin{tikzpicture}\n\\node [draw] at (0,0) {Text};\n\\end{tikzpicture}\n"
  parser2 = TikzParser(source2)
  graph2 = parser2.parse()
  assert len(graph2.nodes) == 0
