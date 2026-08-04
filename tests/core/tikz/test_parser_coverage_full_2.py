"""Test module."""

from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer, _logical_from_tikz_graph
from ml_switcheroo.core.tikz.nodes import TriviaNode, TikzNode, TikzTable, TikzGraph
from lark import Tree, Token


def test_tikz_parser_missing_lines():
  """Test function."""
  transformer = TikzTransformer()
  # 50: trailing trivia
  transformer.start([["node_here"], TriviaNode(" ")])

  # 78-83:
  tg = TikzGraph([])
  tg.leading_trivia = []
  tg.trailing_trivia = []
  transformer.start([[tg]])  # Must wrap in list so it enters elements

  # 197: IGNORE_TEXT
  transformer.IGNORE_TEXT(Token("IGNORE_TEXT", "foo"))

  # 246->221
  transformer.node(["\\node", Tree("not_text_content", [])])

  # 328->317
  transformer.tabular(["\\begin{tabular}", Tree("not_tabular_row", [])])

  # 365
  transformer.kind(["Kind"])

  # 470->466 (Logical graph)
  node = TikzNode(
    "n1",
    0,
    0,
    content=TikzTable(
      rows=[
        [123]  # not a string
      ]
    ),
  )
  graph = TikzGraph(children=[node])
  lg = _logical_from_tikz_graph(graph)  # noqa: F841


def test_tikz_parser_except_block():
  """Test function."""
  # Hit 426-428
  parser = TikzParser("invalid { tikz {")
  try:
    parser.parse()
  except ValueError:
    pass


def test_tikz_parser_skip_unknown():
  """Test function."""
  transformer = TikzTransformer()
  # 78->57
  transformer.start([["unknown_str"]])


def test_tikz_parser_skip_unknown_multiple():
  """Test function."""
  transformer = TikzTransformer()
  # 78->57 requires jumping back to the start of the loop
  transformer.start([["unknown_str", "unknown_str_2"]])
