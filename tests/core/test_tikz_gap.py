import pytest
from ml_switcheroo.core.tikz.parser import TikzParser, TokenKind


def test_tikz_end_command():
  parser = TikzParser(r"\end{tikzpicture}")
  parser.parse()
  # Should complete without error


def test_tikz_peek_eof():
  parser = TikzParser(r"\node (a) {};")
  # offset past end
  token = parser._peek(offset=100)
  assert token.kind == TokenKind.EOF


def test_tikz_expect_error():
  parser = TikzParser(r"\node (a) {};")
  with pytest.raises(SyntaxError):
    parser._expect(TokenKind.LBRACE)


def test_tikz_node_at_coordinates():
  parser = TikzParser(r"\node (a) at (1, 2) {Linear};")
  graph = parser.parse()
  assert len(graph.nodes) == 1


def test_tikz_edge_unexpected_connector():
  # Provide an edge without an arrow
  parser = TikzParser(r"\draw (a) (b);")
  graph = parser.parse()
  assert len(graph.edges) == 0


def test_tikz_scan_until_semicolon_eof():
  parser = TikzParser(r"\draw (a) ")
  # force scan to EOF
  parser._scan_until_semicolon()
  assert parser._is_eof()


def test_tikz_extract_metadata_empty():
  parser = TikzParser(r"\node (a) {\textbf};")
  graph = parser.parse()
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Unknown"


def test_tikz_parser_gaps():
  from ml_switcheroo.core.tikz.parser import TikzParser

  # Line 174: unknown command or unhandled structure
  # In _parse(), if none of the if/elifs match, it falls to self._consume()
  source = r"""
\begin{tikzpicture}
\unknowncommand
\end{tikzpicture}
"""
  parser = TikzParser(source)
  # This shouldn't crash, should just consume and skip
  parser.parse()

  # Line 267-268: node without ID
  # \node [options] at (0,0) {Text}; -> no (id)
  source2 = r"""
\begin{tikzpicture}
\node [draw] at (0,0) {Text};
\end{tikzpicture}
"""
  parser2 = TikzParser(source2)
  graph2 = parser2.parse()
  # It skips it to avoid noise, so nodes should be empty
  assert len(graph2.nodes) == 0
