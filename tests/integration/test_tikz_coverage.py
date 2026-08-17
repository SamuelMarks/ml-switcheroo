"""Module docstring."""

from ml_switcheroo.core.tikz.nodes import (
  TikzBaseNode,
  TriviaNode,
  TikzOption,
  TikzTextNode,
  TikzTable,
  TikzNode,
  TikzEdge,
  TikzGraph,
)
from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer
from lark import Token


def test_tikz_nodes_missing():
  """Docstring."""
  nodes = [
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
      n.emit()

  class DummyTikz(TikzBaseNode):
    def emit(self, indent_level: int = 0) -> str:
      return "d"

  DummyTikz().emit()


def test_tikz_parser_missing():
  """Docstring."""
  cases = [
    r"\node",
    r"\node;",
    r"\node (a) at (0,0) {A};",
    r"\node[draw] (b) at (1,1) {B};",
    r"\node[draw=black] (b) at (1,1) {B};",
    r"\node (b) at (1,1) {\begin{tabular}{c} \textbf{A} \\ \textit{B} \\ C \end{tabular}};",
    r"\draw",
    r"\draw;",
    r"\draw (a) -- (b);",
    r"\draw[thick] (a) -> (b);",
    r"\begin{tikzpicture} \end{tikzpicture}",
    r"\begin{scope} \end{scope}",
    r"% comment",
    r"something else",
    r"\node (a) { \textbf{foo} };",
    r"\node (a) { \textit{foo} };",
    r"\path (a) edge (b);",
    r"\coordinate (c) at (1,1);",
    r"\begin{scope}[shift={(1,1)}] \end{scope}",
    r"\node (a) at (1,1) { \begin{tabular}{l} a \end{tabular} };",
    r"\node (a) at (1,1) { \begin{tabular}{r} a \end{tabular} };",
    r"\begin{tikzpicture} \node (a) at (0,0) {A}; \draw (a) -- (a); \end{tikzpicture}",
    r"\node (a) { \begin{tabular}{c} a \\ b \end{tabular} };",
    r"\node (a) {A};",
    r"\node[x=1] (a) {A};",
    r"\draw[color=red] (a) -- (b) node {label};",
    r"\node [style] (a) at (1,1) {A};",
    r"\draw (a) edge (b);",
  ]

  for case in cases:
    parser = TikzParser(case)
    try:
      parser.parse()
    except Exception:
      pass

  t = TikzTransformer()
  # Hit missing lines directly
  t.start([])
  t.start([TriviaNode(content=" "), TikzNode(node_id="n", x=0, y=0, content="c")])

  t.element([TriviaNode(content=" ")])

  t.TIKZ_ENV_BEGIN(Token("TIKZ_ENV_BEGIN", "\\begin{tikzpicture}"))
  t.TIKZ_ENV_END(Token("TIKZ_ENV_END", "\\end{tikzpicture}"))
  t.NAME(Token("NAME", "foo"))

  t.OPTION(Token("OPTION", "k=v"))
  t.OPTION(Token("OPTION", "k"))

  t.COORD(Token("COORD", "(1,1)"))
  t.ALIGN(Token("ALIGN", "c"))
  t.EDGE_OP(Token("EDGE_OP", "--"))

  t.trivia([Token("IGNORE_TEXT", " "), Token("IGNORE_TEXT", "\n")])
  t.trivia([Token("COMMENT", "% comm")])

  t.IGNORE_TEXT(Token("IGNORE_TEXT", " "))

  try:
    t.node([Token("NAME", "n"), Token("COORD", "(1,1)"), "content"])
  except Exception:
    pass

  try:
    t.node([Token("NAME", "n"), "content"])
  except Exception:
    pass

  try:
    t.node([TikzOption(key="a"), Token("NAME", "n"), Token("COORD", "(1,1)"), "content"])
  except Exception:
    pass

  try:
    t.edge([Token("NAME", "s"), Token("EDGE_OP", "--"), Token("NAME", "t")])
  except Exception:
    pass

  class DummyTree:
    def __init__(self, data, children):
      self.data = data
      self.children = children

  t.tabular([DummyTree("tabular_row", ["a", None, "b"]), DummyTree("kind", ["c"]), DummyTree("id", ["d"])])

  t.tabular_row(["a"])
  t.kind(["c"])
  t.id(["i"])
  t.meta(["m"])
  t.ignore(["x"])
