"""Test suite for the Tikz Nodes module."""

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


def test_trivia_node() -> None:
  """Verifies the behavior of trivia node."""
  node = TriviaNode(content="    ")
  assert node.to_text() == "    "
  newline = TriviaNode(content="\n")
  assert newline.to_text() == "\n"


def test_comment_node() -> None:
  """Verifies the behavior of comment node."""
  c1 = TriviaNode(content="% Hello World\n", kind="comment")  # type: ignore
  assert c1.to_text() == "% Hello World\n"
  c2 = TriviaNode(content="% Already has percent", kind="comment")  # type: ignore
  assert c2.to_text() == "% Already has percent"


def test_option_node() -> None:
  """Verifies the behavior of option node."""
  o1 = TikzOption(key="draw")
  assert o1.to_text() == "draw"
  o2 = TikzOption(key="fill", value="red")
  assert o2.to_text() == "fill=red"


def test_table_node() -> None:
  """Verifies the behavior of table node."""
  table = TikzTable(rows=[["\\textbf{Conv2d}"], ["In: 1", "Out: 32"]])  # type: ignore
  text: str = table.to_text()
  assert "\\begin{tabular}{c}" in text
  assert "\\textbf{Conv2d} \\\\" in text
  assert "In: 1 & Out: 32 \\\\" in text
  assert "\\end{tabular}" in text


def test_node_rendering_simple() -> None:
  """Verifies the behavior of node rendering simple."""
  node = TikzNode(node_id="n1", x=0, y=1.5, content="Start", options=[TikzOption("circle"), TikzOption("draw")])  # type: ignore
  text: str = node.to_text()
  assert "\\node [circle, draw] (n1) at (0, 1.5) {Start};" == text


def test_node_rendering_with_table() -> None:
  """Verifies the behavior of node rendering with table."""
  table = TikzTable(rows=[["Prop"]])  # type: ignore
  node = TikzNode(node_id="n2", x=10, y=20, content=table)  # type: ignore
  text: str = node.to_text()
  assert "\\node (n2) at (10, 20) {" in text
  assert "\\begin{tabular}{c}" in text
  assert "};" in text


def test_edge_rendering() -> None:
  """Verifies the behavior of edge rendering."""
  edge = TikzEdge(source_id="a", target_id="b", options=[TikzOption("->"), TikzOption("thick")])
  text: str = edge.to_text()
  assert "\\draw [->, thick] (a) -- (b);" == text


def test_edge_rendering_with_trivia() -> None:
  """Verifies the behavior of edge rendering with trivia."""
  edge = TikzEdge(source_id="a", target_id="b", leading_trivia=[TriviaNode("\n    ")])  # type: ignore
  text: str = edge.to_text()
  assert "\n    \\draw" in text


def test_graph_composition() -> None:
  """Verifies the behavior of graph composition."""
  node1 = TikzNode(node_id="a", x=0, y=0, content="A")  # type: ignore
  node2 = TikzNode(node_id="b", x=1, y=0, content="B")  # type: ignore
  edge = TikzEdge(source_id="a", target_id="b")
  graph = TikzGraph(
    options=[TikzOption("scale", "0.5")],
    children=[TriviaNode("\n% Nodes\n"), node1, node2, TriviaNode("\n"), TriviaNode("% Edges\n"), edge, TriviaNode("\n")],  # type: ignore
  )
  text: str = graph.to_text()
  assert "\\begin{tikzpicture}[scale=0.5]" in text
  assert "\\end{tikzpicture}" in text
  lines: list[str] = text.splitlines()
  assert lines[1].strip() == "% Nodes"
  assert "\\node (a) at (0, 0) {A};" in text
  assert "\\draw (a) -- (b);" in text


# --- Merged from test_tikz_nodes_extra.py ---


def test_base_node():
  """Docstring."""

  class DummyNode(TikzBaseNode):
    """A dummy node."""

    pass

  node = DummyNode()
  assert node.emit() == ""
  assert node.to_text() == ""


def test_trivia_node_extra():
  """Docstring."""
  node = TriviaNode(content="  ", kind="whitespace")
  assert node.emit() == "  "


def test_tikz_option():
  """Docstring."""
  opt = TikzOption(key="draw")
  assert opt.emit() == "draw"
  opt = TikzOption(key="draw", value="black")
  assert opt.emit() == "draw=black"


def test_tikz_text_node():
  """Docstring."""
  node = TikzTextNode(content="Hello")
  assert node.emit() == "Hello"
  node = TikzTextNode(content="Hello", bold=True)
  assert node.emit() == "\\textbf{Hello}"
  node = TikzTextNode(content="Hello", italic=True)
  assert node.emit() == "\\textit{Hello}"
  node = TikzTextNode(content="Hello", bold=True, italic=True)
  assert node.emit() == "\\textbf{\\textit{Hello}}"


def test_tikz_table():
  """Docstring."""
  row = ["plain text", TikzTextNode("bold text", bold=True)]
  table = TikzTable(rows=[row], align="c", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode("\n")])
  res = table.emit()
  assert res == " \\begin{tabular}{c}plain text & \\textbf{bold text} \\\\\\end{tabular}\n"


def test_tikz_node():
  """Docstring."""
  node = TikzNode(
    node_id="n1",
    x=1.0,
    y=2.0,
    content="label",
    options=[TikzOption("draw")],
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode("\n")],
  )
  res = node.emit()
  assert res == " \\node [draw] (n1) at (1.0, 2.0) {label};\n"

  table_content = TikzTable(rows=[["A"]])
  node2 = TikzNode(node_id="n2", x=0.0, y=0.0, content=table_content)
  assert "\\begin{tabular}" in node2.emit()


def test_tikz_edge():
  """Docstring."""
  edge = TikzEdge(
    source_id="n1",
    target_id="n2",
    options=[TikzOption("->")],
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode("\n")],
  )
  res = edge.emit()
  assert res == " \\draw [->] (n1) -- (n2);\n"


def test_tikz_graph():
  """Docstring."""
  graph = TikzGraph(
    children=[TikzNode(node_id="n1", x=0.0, y=0.0, content="A")],
    options=[TikzOption("scale", "2")],
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode("\n")],
  )
  res = graph.emit()
  assert res == " \\begin{tikzpicture}[scale=2]\\node (n1) at (0.0, 0.0) {A};\\end{tikzpicture}\n"

  graph2 = TikzGraph(children=[TikzNode(node_id="n1", x=0.0, y=0.0, content="A")])
  res2 = graph2.emit()
  assert "\\begin{tikzpicture}\\node (n1) at (0.0, 0.0) {A};\\end{tikzpicture}" == res2
