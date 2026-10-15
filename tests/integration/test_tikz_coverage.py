"""Test suite for tikz coverage."""

import typing

import pytest
from lark import Token
from lark.exceptions import LarkError

from ml_switcheroo.core.tikz.nodes import (
  TikzBaseNode,
  TikzEdge,
  TikzGraph,
  TikzNode,
  TikzOption,
  TikzTable,
  TikzTextNode,
  TriviaNode,
)
from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer


def test_tikz_nodes_missing() -> None:
  """Verify tikz nodes missing behavior."""
  nodes: list[typing.Any] = [
    TriviaNode(content=" "),
    TikzOption(key="a", value="b"),
    TikzOption(key="c"),
    TikzTextNode(content="c"),
    TikzTextNode(content="c", bold=True, italic=True),
    TikzTable(rows=[["a", "b"]], align="l", leading_trivia=[TriviaNode(content=" ")]),
    TikzTable(rows=[["a", "b"]], align="r", leading_trivia=[TriviaNode(content=" ")]),
    TikzNode(node_id="n", x=0.0, y=0.0, content="l", options=[TikzOption(key="d")]),
    TikzNode(node_id="n", x=0.0, y=0.0, content=TikzTable(rows=[["a", "b"]]), options=[TikzOption(key="d")]),
    TikzEdge(source_id="s", target_id="t", options=[]),
    TikzEdge(source_id="s", target_id="t", options=[], connector="->"),
    TikzGraph(children=[]),
  ]
  for n in nodes:
    assert repr(n)
    if hasattr(n, "emit"):
      assert isinstance(n.emit(), str)

  class DummyTikz(TikzBaseNode):  # type: ignore[misc]
    """Test helper class DummyTikz."""

    def emit(self, indent_level: int = 0) -> str:
      """Execute emit helper logic."""
      return "d"

  assert DummyTikz().emit() == "d"


VALID_TIKZ_CASES = [
  r"\node (a) at (0,0) {A};",
  r"\node[draw] (b) at (1,1) {B};",
  r"\node[draw=black] (b) at (1,1) {B};",
  r"\draw (a) -- (b);",
  r"\draw[thick] (a) -> (b);",
  r"\begin{tikzpicture} \end{tikzpicture}",
  r"\begin{tikzpicture} \node (a) at (0,0) {A}; \draw (a) -- (a); \end{tikzpicture}",
  r"\node (a) {A};",
  r"\node[x=1] (a) {A};",
  r"\node [style] (a) at (1,1) {A};",
]

INVALID_TIKZ_CASES = [
  r"\node",
  r"\node;",
  r"\draw",
  r"\draw;",
]


@pytest.mark.parametrize("case", VALID_TIKZ_CASES)
def test_valid_tikz_parsing(case: str) -> None:
  """Verifies that valid Tikz cases parse into a valid tree."""
  parser = TikzParser(case)
  tree = parser.parse()
  assert tree is not None
  assert isinstance(tree, TikzBaseNode)
  # Emission should not crash
  assert isinstance(tree.emit(), str)


@pytest.mark.parametrize("case", INVALID_TIKZ_CASES)
def test_invalid_tikz_parsing(case: str) -> None:
  """Verifies that invalid Tikz cases correctly throw parsing exceptions."""
  parser = TikzParser(case)
  with pytest.raises((LarkError, ValueError)):
    parser.parse()


def test_transformer_explicit() -> None:
  """Verifies transformer methods directly without exception swallowing."""
  t = TikzTransformer()
  # Hit missing lines directly
  assert t.start([]) is not None
  assert t.start([TriviaNode(content=" "), TikzNode(node_id="n", x=0, y=0, content="c")]) is not None

  assert t.element([TriviaNode(content=" ")]) is not None

  assert t.TIKZ_ENV_BEGIN(Token("TIKZ_ENV_BEGIN", "\\begin{tikzpicture}")) is not None
  assert t.TIKZ_ENV_END(Token("TIKZ_ENV_END", "\\end{tikzpicture}")) is not None
  assert t.NAME(Token("NAME", "foo")) is not None

  assert t.OPTION(Token("OPTION", "k=v")) is not None
  assert t.OPTION(Token("OPTION", "k")) is not None

  assert t.COORD(Token("COORD", "(1,1)")) is not None
  assert t.ALIGN(Token("ALIGN", "c")) is not None
  assert t.EDGE_OP(Token("EDGE_OP", "--")) is not None

  assert t.trivia([Token("IGNORE_TEXT", " "), Token("IGNORE_TEXT", "\n")]) is not None
  assert t.trivia([Token("COMMENT", "% comm")]) is not None

  assert t.IGNORE_TEXT(Token("IGNORE_TEXT", " ")) is not None

  # These transformer methods expect specific sequences.
  res_node = t.node([Token("NAME", "n"), Token("COORD", "(1,1)"), "content"])
  assert isinstance(res_node, TikzNode)

  res_node2 = t.node([Token("NAME", "n"), "content"])
  assert isinstance(res_node2, TikzNode)

  res_node3 = t.node([TikzOption(key="a"), Token("NAME", "n"), Token("COORD", "(1,1)"), "content"])
  assert isinstance(res_node3, TikzNode)

  res_edge = t.edge([Token("NAME", "s"), Token("EDGE_OP", "--"), Token("NAME", "t")])
  assert isinstance(res_edge, TikzEdge)

  class DummyTree:
    """Execute DummyTree test helper."""

    def __init__(self, data: str, children: list[typing.Any]) -> None:
      """Execute __init__ test helper."""
      self.data: str = data
      self.children: list[typing.Any] = children

  res_tab = t.tabular([DummyTree("tabular_row", ["a", None, "b"]), DummyTree("kind", ["c"]), DummyTree("id", ["d"])])
  assert isinstance(res_tab, TikzTable)

  assert t.tabular_row(["a"]) is not None
  assert t.kind(["c"]) is not None
  assert t.id(["i"]) is not None
  assert t.meta(["m"]) is not None
  assert t.ignore(["x"]) is not None
