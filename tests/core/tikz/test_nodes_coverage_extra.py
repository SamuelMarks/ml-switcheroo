"""Test module."""

from ml_switcheroo.core.tikz.nodes import TikzTextNode, TikzTable, TikzNode, TikzEdge, TikzGraph, TriviaNode, TikzOption


def test_tikz_text_node():
  """Test function."""
  node = TikzTextNode(content="hello", bold=True, italic=True)
  assert node.emit() == "\\textbf{\\textit{hello}}"


def test_tikz_table_full():
  """Test function."""
  t1 = TriviaNode(content=" % lead\n")
  t2 = TriviaNode(content=" % trail\n")
  tbl = TikzTable(rows=[[TikzTextNode(content="A", bold=True), "B"]], leading_trivia=[t1], trailing_trivia=[t2])
  res = tbl.emit()
  assert "% lead" in res
  assert "% trail" in res
  assert "\\textbf{A} & B" in res


def test_tikz_node_full():
  """Test function."""
  t1 = TriviaNode(content=" % nlead\n")
  t2 = TriviaNode(content=" % ntrail\n")
  n = TikzNode(
    node_id="n1",
    x=1.0,
    y=2.0,
    content=TikzTextNode("C"),
    options=[TikzOption(key="draw")],
    leading_trivia=[t1],
    trailing_trivia=[t2],
  )
  res = n.emit()
  assert "% nlead" in res
  assert "% ntrail" in res
  assert "\\node" in res
  assert "C" in res


def test_tikz_edge_full():
  """Test function."""
  t1 = TriviaNode(content=" % elead\n")
  t2 = TriviaNode(content=" % etrail\n")
  e = TikzEdge(source_id="A", target_id="B", options=[TikzOption(key="thick")], leading_trivia=[t1], trailing_trivia=[t2])
  res = e.emit()
  assert "% elead" in res
  assert "% etrail" in res
  assert "\\draw" in res


def test_tikz_graph_full():
  """Test function."""
  t1 = TriviaNode(content=" % glead\n")
  t2 = TriviaNode(content=" % gtrail\n")
  g = TikzGraph(children=[], options=[TikzOption(key="scale", value="2")], leading_trivia=[t1], trailing_trivia=[t2])
  res = g.emit()
  assert "% glead" in res
  assert "% gtrail" in res
  assert "\\begin{tikzpicture}[scale=2]" in res
