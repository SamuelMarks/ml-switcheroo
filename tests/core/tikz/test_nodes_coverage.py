"""Tests for TikZ nodes coverage."""

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


def test_tikz_base_node_abstract() -> None:
  """Docstring."""

  class Dummy(TikzBaseNode):
    """Dummy."""

    def to_text(self) -> str:
      """To text."""
      try:
        super().to_text()
      except NotImplementedError:
        pass
      return ""

  assert Dummy().to_text() == ""


def test_nodenode_with_trivia() -> None:
  """Docstring."""
  triv = TriviaNode("% comment")
  node = TikzNode(node_id="A", x=0.0, y=0.0, leading_trivia=[triv], options=[], content="")  # type: ignore
  res: str = node.to_text()
  assert "% comment" in res


def test_tikzpicturenode_no_options() -> None:
  """Docstring."""
  pic = TikzGraph(options=[], children=[])
  res: str = pic.to_text()
  assert "\\begin{tikzpicture}" in res


# --- Merged from test_nodes_coverage_extra.py ---


def test_tikz_text_node() -> None:
  """Docstring."""
  node = TikzTextNode(content="hello", bold=True, italic=True)
  assert node.emit() == "\\textbf{\\textit{hello}}"


def test_tikz_table_full() -> None:
  """Docstring."""
  t1 = TriviaNode(content=" % lead\n")
  t2 = TriviaNode(content=" % trail\n")
  tbl = TikzTable(rows=[[TikzTextNode(content="A", bold=True), "B"]], leading_trivia=[t1], trailing_trivia=[t2])  # type: ignore
  res: str = tbl.emit()
  assert "% lead" in res
  assert "% trail" in res
  assert "\\textbf{A} & B" in res


def test_tikz_node_full() -> None:
  """Docstring."""
  t1 = TriviaNode(content=" % nlead\n")
  t2 = TriviaNode(content=" % ntrail\n")
  n = TikzNode(
    node_id="n1",
    x=1.0,
    y=2.0,
    content=TikzTextNode("C"),
    options=[TikzOption(key="draw")],
    leading_trivia=[t1],  # type: ignore
    trailing_trivia=[t2],  # type: ignore
  )
  res: str = n.emit()
  assert "% nlead" in res
  assert "% ntrail" in res
  assert "\\node" in res
  assert "C" in res


def test_tikz_edge_full() -> None:
  """Docstring."""
  t1 = TriviaNode(content=" % elead\n")
  t2 = TriviaNode(content=" % etrail\n")
  e = TikzEdge(source_id="A", target_id="B", options=[TikzOption(key="thick")], leading_trivia=[t1], trailing_trivia=[t2])  # type: ignore
  res: str = e.emit()
  assert "% elead" in res
  assert "% etrail" in res
  assert "\\draw" in res


def test_tikz_graph_full() -> None:
  """Docstring."""
  t1 = TriviaNode(content=" % glead\n")
  t2 = TriviaNode(content=" % gtrail\n")
  g = TikzGraph(children=[], options=[TikzOption(key="scale", value="2")], leading_trivia=[t1], trailing_trivia=[t2])  # type: ignore
  res: str = g.emit()
  assert "% glead" in res
  assert "% gtrail" in res
  assert "\\begin{tikzpicture}[scale=2]" in res
