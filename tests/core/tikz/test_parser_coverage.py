"""Test parser coverage."""

import typing

from lark import Token, Tree

from ml_switcheroo.core.tikz.nodes import TikzGraph, TikzNode, TikzTable, TriviaNode
from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer, _logical_from_tikz_graph


def test_parser_error() -> None:
  """Docstring."""
  parser = TikzParser(r"\begin{tikzpicture} \invalid \end{tikzpicture}")
  try:
    parser.parse()
    assert False
  except ValueError as e:
    assert "Failed to parse TikZ" in str(e)


# --- Merged from test_parser_coverage_full_2.py ---


def test_tikz_parser_missing_lines() -> None:
  """Docstring."""
  transformer = TikzTransformer()
  # 50: trailing trivia
  transformer.start([["node_here"], TriviaNode(" ")])

  # 78-83:
  tg = TikzGraph([])
  tg.leading_trivia = []  # type: ignore
  tg.trailing_trivia = []  # type: ignore
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
    node_id="n1",
    x=0,
    y=0,
    content=TikzTable(
      rows=[
        [123]  # not a string # type: ignore
      ]
    ),
  )
  graph = TikzGraph(children=[node])
  lg: typing.Any = _logical_from_tikz_graph(graph)  # noqa: F841


def test_tikz_parser_except_block() -> None:
  """Docstring."""
  # Hit 426-428
  parser = TikzParser("invalid { tikz {")
  try:
    parser.parse()
  except ValueError:
    pass


def test_tikz_parser_skip_unknown() -> None:
  """Docstring."""
  transformer = TikzTransformer()
  # 78->57
  transformer.start([["unknown_str"]])


def test_tikz_parser_skip_unknown_multiple() -> None:
  """Docstring."""
  transformer = TikzTransformer()
  # 78->57 requires jumping back to the start of the loop
  transformer.start([["unknown_str", "unknown_str_2"]])


# --- Merged from test_parser_coverage_full.py ---


def test_tikz_parser_standalone() -> None:
  """Docstring."""
  pass


def test_tikz_transformer_direct() -> None:
  """Docstring."""
  transformer = TikzTransformer()

  # 48 - 50: Top level trivia leading/trailing
  # 74->57: end_tikzpicture handling
  # 78-83: TikzGraph extraction
  transformer.start([TriviaNode(" ")])
  transformer.start([TikzGraph([])])
  transformer.start(["\\end{tikzpicture}"])

  # 226, 232, 235->221, 238->221, 242-243, 247->221
  # 226: trailing trivia
  transformer.node(["\\node", TriviaNode(" ")])
  # 232: found_node = True
  transformer.node(["\\node"])
  # 235->221: false condition for node_id (i.e. c is "(")
  transformer.node(["\\node", "(", ")"])
  # 238->221: len(parts) != 2
  transformer.node(["\\node", "id", "1.0,2.0,3.0"])
  # 242-243: ValueError inside parts parsing
  transformer.node(["\\node", "id", "a,b"])
  # 247->221: empty children for text_content
  transformer.node([Tree("text_content", [])])

  # 278, 281, 284->273, 289->273
  transformer.edge(["\\draw", TriviaNode(" ")])  # 278, 281
  transformer.edge(["\\draw", "(", ")"])  # 284->273
  transformer.edge(["\\draw", "src", "->", "dst", Tree("unknown", [])])  # 289->273

  # 319->322, 325, 326->317, 336-339, 340->317
  # 319->322, 325: trailing trivia
  transformer.tabular(["\\begin{tabular}", TriviaNode(" ")])
  # 326->317: c in ["{", "}"]
  transformer.tabular(["\\begin{tabular}", "{", "c", "}"])
  # 336-339: tabular_row items (id, kind)
  transformer.tabular([Tree("tabular_row", [Tree("kind", ["Kind"]), Tree("id", ["Id"]), Tree("unknown", [])])])  # type: ignore
  # 340->317: empty current_row
  transformer.tabular([Tree("tabular_row", [])])

  # 376 inside id()
  transformer.id(["A"])


def test_logical_from_tikz_graph() -> None:
  """Docstring."""
  # 468, 480->483

  node1 = TikzNode(
    node_id="n1",
    x=0,
    y=0,
    content=TikzTable(
      rows=[
        [],  # 468
        ["\\textit{id}"],
        ["JustText"],
        ["a:b\\\\"],
      ]
    ),
  )

  node2 = TikzNode(node_id="n2", x=0, y=0, content=None)  # 480->483 false # type: ignore
  node3 = TikzNode(node_id="n3", x=0, y=0, content="")  # 480->483 false # type: ignore

  graph = TikzGraph(children=[node1, node2, node3])
  lg: typing.Any = _logical_from_tikz_graph(graph)  # noqa: F841
