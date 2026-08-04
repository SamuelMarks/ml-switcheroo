"""Test module."""

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


def test_base_node():
  """Test function."""

  class DummyNode(TikzBaseNode):
    pass

  node = DummyNode()
  assert node.emit() == ""
  assert node.to_text() == ""


def test_trivia_node():
  """Test function."""
  node = TriviaNode(content="  ", kind="whitespace")
  assert node.emit() == "  "


def test_tikz_option():
  """Test function."""
  opt = TikzOption(key="draw")
  assert opt.emit() == "draw"
  opt = TikzOption(key="draw", value="black")
  assert opt.emit() == "draw=black"


def test_tikz_text_node():
  """Test function."""
  node = TikzTextNode(content="Hello")
  assert node.emit() == "Hello"
  node = TikzTextNode(content="Hello", bold=True)
  assert node.emit() == "\\textbf{Hello}"
  node = TikzTextNode(content="Hello", italic=True)
  assert node.emit() == "\\textit{Hello}"
  node = TikzTextNode(content="Hello", bold=True, italic=True)
  assert node.emit() == "\\textbf{\\textit{Hello}}"


def test_tikz_table():
  """Test function."""
  row = ["plain text", TikzTextNode("bold text", bold=True)]
  table = TikzTable(rows=[row], align="c", leading_trivia=[TriviaNode(" ")], trailing_trivia=[TriviaNode("\n")])
  res = table.emit()
  assert res == " \\begin{tabular}{c}plain text & \\textbf{bold text} \\\\\\end{tabular}\n"


def test_tikz_node():
  """Test function."""
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
  """Test function."""
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
  """Test function."""
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
