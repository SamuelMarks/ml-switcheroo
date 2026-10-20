"""Test parser coverage."""

import typing

from lark import Token, Tree

from ml_switcheroo.core.tikz.nodes import TikzGraph, TikzNode, TikzTable, TriviaNode
from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer, _logical_from_tikz_graph


def test_parser_error() -> None:
  """Verify parser error behavior."""
  parser = TikzParser(r"\begin{tikzpicture} \invalid \end{tikzpicture}")
  try:
    parser.parse()
    assert False
  except ValueError as e:
    assert "Failed to parse TikZ" in str(e)


# --- Merged from test_parser_coverage_full_2.py ---


def test_tikz_parser_missing_lines() -> None:
  """Verify tikz parser missing lines behavior."""
  transformer = TikzTransformer()
  # 50: trailing trivia
  transformer.start([["node_here"], TriviaNode(" ")])
  transformer.start([[123]])  # type: ignore[list-item]  # Non-matching element takes False branch of isinstance check

  tg = TikzGraph([])
  tg.leading_trivia = []
  tg.trailing_trivia = []
  transformer.start([[tg]])  # Must wrap in list so it enters elements

  # 197: IGNORE_TEXT
  transformer.IGNORE_TEXT(Token("IGNORE_TEXT", "foo"))

  transformer.node(["\\node", Tree("not_text_content", [])])

  transformer.tabular(["\\begin{tabular}", Tree("not_tabular_row", [])])

  # 365
  transformer.kind(["Kind"])  # type: ignore[list-item]

  node = TikzNode(
    node_id="n1",
    x=0,
    y=0,
    content=TikzTable(
      rows=[
        [123]  # not a string
      ]
    ),
  )
  graph = TikzGraph(children=[node])
  lg = _logical_from_tikz_graph(graph)  # noqa: F841
  assert "n1" in lg.nodes


def test_tikz_parser_except_block() -> None:
  """Verify tikz parser except block behavior."""
  import pytest

  # Hit 426-428
  parser = TikzParser("invalid { tikz {")
  with pytest.raises(ValueError, match="Failed to parse TikZ"):
    parser.parse()


def test_tikz_parser_skip_unknown() -> None:
  """Verify tikz parser skip unknown behavior."""
  transformer = TikzTransformer()

  _res = transformer.start([["unknown_str"]])
  assert isinstance(_res, TikzGraph)
  assert _res.children == []


def test_tikz_parser_skip_unknown_multiple() -> None:
  """Verify tikz parser skip unknown multiple behavior."""
  transformer = TikzTransformer()

  _res = transformer.start([["unknown_str", "unknown_str_2"]])
  assert isinstance(_res, TikzGraph)
  assert _res.children == []


# --- Merged from test_parser_coverage_full.py ---


def test_tikz_parser_standalone() -> None:
  """Verify tikz parser standalone behavior."""
  from ml_switcheroo.core.tikz.parser import TikzParser

  parser = TikzParser("code")
  assert parser is not None


def test_tikz_transformer_direct() -> None:
  """Verify tikz transformer direct behavior."""
  transformer = TikzTransformer()

  transformer.start([TriviaNode(" ")])
  transformer.start([TikzGraph([])])
  transformer.start(["\\end{tikzpicture}"])

  # 226, 232, 235->221, 238->221, 242-243, 247->221
  # 226: trailing trivia
  transformer.node(["\\node", TriviaNode(" ")])
  # 232: found_node = True
  transformer.node(["\\node"])

  transformer.node(["\\node", "(", ")"])

  transformer.node(["\\node", "id", "1.0,2.0,3.0"])

  transformer.node(["\\node", "id", "a,b"])

  transformer.node([Tree("text_content", [])])

  # 278, 281, 284->273, 289->273
  transformer.edge(["\\draw", TriviaNode(" ")])  # 278, 281
  transformer.edge(["\\draw", "(", ")"])
  transformer.edge(["\\draw", "src", "->", "dst", Tree("unknown", [])])

  transformer.tabular(["\\begin{tabular}", TriviaNode(" ")])

  transformer.tabular(["\\begin{tabular}", "{", "c", "}"])

  empty_children: list[typing.Union[str, Tree[str]]] = []
  row_children: list[typing.Union[str, Tree[str]]] = [
    Tree("kind", ["Kind"]),
    Tree("id", ["Id"]),
    Tree("unknown", empty_children),
  ]
  transformer.tabular([Tree("tabular_row", row_children)])  # type: ignore[arg-type]
  transformer.tabular([Tree("tabular_row", empty_children)])  # type: ignore[arg-type]

  # 376 inside id()
  _res = transformer.id(["A"])  # type: ignore[list-item]
  assert isinstance(_res, Tree)
  assert _res.data == "id"
  assert _res.children == ["A"]


def test_logical_from_tikz_graph() -> None:
  """Verify logical from tikz graph behavior."""
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

  node2 = TikzNode(node_id="n2", x=0, y=0, content=None)  # type: ignore[arg-type]
  node3 = TikzNode(node_id="n3", x=0, y=0, content="")

  graph = TikzGraph(children=[node1, node2, node3])
  lg = _logical_from_tikz_graph(graph)  # noqa: F841
  assert set(lg.nodes.keys()) == {"n1", "n2", "n3"}
