"""Tests for TikZ nodes coverage."""

import typing

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
  """Verify tikz base node abstract behavior."""

  class Dummy(TikzBaseNode):
    """Test helper class Dummy."""

    def to_text(self) -> typing.Any:
      """To text.

      Returns:
          Computed test result or fixture value.
      """
      try:
        super().to_text()
      except NotImplementedError:
        pass
      return ""

  assert Dummy().to_text() == ""


def test_nodenode_with_trivia() -> None:
  """Verify nodenode with trivia behavior."""
  triv: typing.Any = TriviaNode("% comment")
  node: typing.Any = TikzNode(node_id="A", x=0.0, y=0.0, leading_trivia=[triv], options=[], content="")
  res: typing.Any = node.to_text()
  assert "% comment" in res


def test_tikzpicturenode_no_options() -> None:
  """Verify tikzpicturenode no options behavior."""
  pic: typing.Any = TikzGraph(options=[], children=[])
  res: typing.Any = pic.to_text()
  assert "\\begin{tikzpicture}" in res


# --- Merged from test_nodes_coverage_extra.py ---


def test_tikz_text_node() -> None:
  """Verify tikz text node behavior."""
  node: typing.Any = TikzTextNode(content="hello", bold=True, italic=True)
  assert node.emit() == "\\textbf{\\textit{hello}}"


def test_tikz_table_full() -> None:
  """Verify tikz table full behavior."""
  t1: typing.Any = TriviaNode(content=" % lead\n")
  t2: typing.Any = TriviaNode(content=" % trail\n")
  tbl: typing.Any = TikzTable(
    rows=[[TikzTextNode(content="A", bold=True), "B"]], leading_trivia=[t1], trailing_trivia=[t2]
  )
  res: typing.Any = tbl.emit()
  assert "% lead" in res
  assert "% trail" in res
  assert "\\textbf{A} & B" in res


def test_tikz_node_full() -> None:
  """Verify tikz node full behavior."""
  t1: typing.Any = TriviaNode(content=" % nlead\n")
  t2: typing.Any = TriviaNode(content=" % ntrail\n")
  n: typing.Any = TikzNode(
    node_id="n1",
    x=1.0,
    y=2.0,
    content=TikzTextNode("C"),
    options=[TikzOption(key="draw")],
    leading_trivia=[t1],
    trailing_trivia=[t2],
  )
  res: typing.Any = n.emit()
  assert "% nlead" in res
  assert "% ntrail" in res
  assert "\\node" in res
  assert "C" in res


def test_tikz_edge_full() -> None:
  """Verify tikz edge full behavior."""
  t1: typing.Any = TriviaNode(content=" % elead\n")
  t2: typing.Any = TriviaNode(content=" % etrail\n")
  e: typing.Any = TikzEdge(
    source_id="A", target_id="B", options=[TikzOption(key="thick")], leading_trivia=[t1], trailing_trivia=[t2]
  )
  res: typing.Any = e.emit()
  assert "% elead" in res
  assert "% etrail" in res
  assert "\\draw" in res


def test_tikz_graph_full() -> None:
  """Verify tikz graph full behavior."""
  t1: typing.Any = TriviaNode(content=" % glead\n")
  t2: typing.Any = TriviaNode(content=" % gtrail\n")
  g: typing.Any = TikzGraph(
    children=[], options=[TikzOption(key="scale", value="2")], leading_trivia=[t1], trailing_trivia=[t2]
  )
  res: typing.Any = g.emit()
  assert "% glead" in res
  assert "% gtrail" in res
  assert "\\begin{tikzpicture}[scale=2]" in res
