"""Test suite for the Tikz Nodes module."""

from ml_switcheroo.core.tikz.nodes import TriviaNode, TikzComment, TikzOption, TikzTable, TikzNode, TikzEdge, TikzGraph


def test_trivia_node():
  """Verifies the behavior of trivia node."""
  node = TriviaNode(content="    ")
  assert node.to_text() == "    "
  newline = TriviaNode(content="\n")
  assert newline.to_text() == "\n"


def test_comment_node():
  """Verifies the behavior of comment node."""
  c1 = TikzComment(text="Hello World")
  assert c1.to_text() == "% Hello World\n"
  c2 = TikzComment(text="% Already has percent", trailing_newline=False)
  assert c2.to_text() == "% Already has percent"


def test_option_node():
  """Verifies the behavior of option node."""
  o1 = TikzOption(key="draw")
  assert o1.to_text() == "draw"
  o2 = TikzOption(key="fill", value="red")
  assert o2.to_text() == "fill=red"


def test_table_node():
  """Verifies the behavior of table node."""
  table = TikzTable(rows=[["\\textbf{Conv2d}"], ["In: 1", "Out: 32"]])
  text = table.to_text()
  assert "\\begin{tabular}{c}" in text
  assert "\\textbf{Conv2d} \\\\" in text
  assert "In: 1 & Out: 32 \\\\" in text
  assert "\\end{tabular}" in text


def test_node_rendering_simple():
  """Verifies the behavior of node rendering simple."""
  node = TikzNode(node_id="n1", x=0, y=1.5, content="Start", options=[TikzOption("circle"), TikzOption("draw")])
  text = node.to_text()
  assert "\\node [circle, draw] (n1) at (0, 1.5) {Start};" == text


def test_node_rendering_with_table():
  """Verifies the behavior of node rendering with table."""
  table = TikzTable(rows=[["Prop"]])
  node = TikzNode(node_id="n2", x=10, y=20, content=table)
  text = node.to_text()
  assert "\\node (n2) at (10, 20) {" in text
  assert "\\begin{tabular}{c}" in text
  assert "};" in text


def test_edge_rendering():
  """Verifies the behavior of edge rendering."""
  edge = TikzEdge(source_id="a", target_id="b", options=[TikzOption("->"), TikzOption("thick")])
  text = edge.to_text()
  assert "\\draw [->, thick] (a) -- (b);" == text


def test_edge_rendering_with_trivia():
  """Verifies the behavior of edge rendering with trivia."""
  edge = TikzEdge(source_id="a", target_id="b", leading_trivia=[TriviaNode("\n    ")])
  text = edge.to_text()
  assert "\n    \\draw" in text


def test_graph_composition():
  """Verifies the behavior of graph composition."""
  node1 = TikzNode(node_id="a", x=0, y=0, content="A")
  node2 = TikzNode(node_id="b", x=1, y=0, content="B")
  edge = TikzEdge(source_id="a", target_id="b")
  graph = TikzGraph(
    options=[TikzOption("scale", "0.5")],
    children=[TikzComment("Nodes"), node1, node2, TriviaNode("\n"), TikzComment("Edges"), edge],
  )
  text = graph.to_text()
  assert "\\begin{tikzpicture}[scale=0.5]" in text
  assert "\\end{tikzpicture}" in text
  lines = text.splitlines()
  assert lines[1].strip() == "% Nodes"
  assert lines[1].startswith("    ")
  assert "\\node (a) at (0, 0) {A};" in text
  assert "\\draw (a) -- (b);" in text
