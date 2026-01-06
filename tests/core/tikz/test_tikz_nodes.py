"""
Tests for TikZ CST Nodes.

Verifies:
1.  Correct string rendering (`to_text`) for all node types.
2.  Proper handling of formatting, indentation, and trivia.
3.  Composition of complex graphs.
"""

from ml_switcheroo.core.tikz.nodes import (
  TriviaNode,
  TikzComment,
  TikzOption,
  TikzTable,
  TikzNode,
  TikzEdge,
  TikzGraph,
)


def test_trivia_node():
  """Verify whitespace preservation."""
  node = TriviaNode(content="    ")
  assert node.to_text() == "    "
  newline = TriviaNode(content="\n")
  assert newline.to_text() == "\n"


def test_comment_node():
  """Verify comment formatting."""
  c1 = TikzComment(text="Hello World")
  assert c1.to_text() == "% Hello World\n"

  c2 = TikzComment(text="% Already has percent", trailing_newline=False)
  assert c2.to_text() == "% Already has percent"


def test_option_node():
  """Verify option formatting."""
  o1 = TikzOption(key="draw")
  assert o1.to_text() == "draw"

  o2 = TikzOption(key="fill", value="red")
  assert o2.to_text() == "fill=red"


def test_table_node():
  """Verify tabular environment construction."""
  table = TikzTable(
    rows=[
      [r"\textbf{Conv2d}"],
      ["In: 1", "Out: 32"],
    ]
  )
  text = table.to_text()

  assert r"\begin{tabular}{c}" in text
  assert r"\textbf{Conv2d} \\" in text
  assert r"In: 1 & Out: 32 \\" in text
  assert r"\end{tabular}" in text


def test_node_rendering_simple():
  """Verify simple node construction."""
  node = TikzNode(
    node_id="n1",
    x=0,
    y=1.5,
    content="Start",
    options=[TikzOption("circle"), TikzOption("draw")],
  )
  text = node.to_text()
  assert r"\node [circle, draw] (n1) at (0, 1.5) {Start};" == text


def test_node_rendering_with_table():
  """Verify node with nested table content."""
  table = TikzTable(rows=[["Prop"]])
  node = TikzNode(
    node_id="n2",
    x=10,
    y=20,
    content=table,
  )
  text = node.to_text()

  assert r"\node (n2) at (10, 20) {" in text
  assert r"\begin{tabular}{c}" in text
  assert "};" in text


def test_edge_rendering():
  """Verify draw command construction."""
  edge = TikzEdge(
    source_id="a",
    target_id="b",
    options=[TikzOption("->"), TikzOption("thick")],
  )
  text = edge.to_text()
  assert r"\draw [->, thick] (a) -- (b);" == text


def test_edge_rendering_with_trivia():
  """Verify leading trivia (indentation/comments) on edges."""
  edge = TikzEdge(
    source_id="a",
    target_id="b",
    leading_trivia=[TriviaNode("\n    ")],
  )
  text = edge.to_text()
  assert r"\n    \draw" in text


def test_graph_composition():
  """
  Verify full graph structure generation with indentation.
  """
  node1 = TikzNode(node_id="a", x=0, y=0, content="A")
  node2 = TikzNode(node_id="b", x=1, y=0, content="B")
  edge = TikzEdge(source_id="a", target_id="b")

  graph = TikzGraph(
    options=[TikzOption("scale", "0.5")],
    children=[
      TikzComment("Nodes"),
      node1,
      node2,
      TriviaNode("\n"),
      TikzComment("Edges"),
      edge,
    ],
  )

  text = graph.to_text()

  # Check Environment
  assert r"\begin{tikzpicture}[scale=0.5]" in text
  assert r"\end{tikzpicture}" in text

  # Check Indentation of children
  lines = text.splitlines()
  # The first line is begin{tikzpicture}, next should be indented
  assert lines[1].strip() == r"% Nodes"
  # Ensure it starts with spaces
  assert lines[1].startswith("    ")

  # Check content presence
  assert r"\node (a) at (0, 0) {A};" in text
  assert r"\draw (a) -- (b);" in text
