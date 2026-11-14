"""Unit tests for TikZ CST nodes (`ml_switcheroo.core.tikz.nodes`)."""

from ml_switcheroo.core.tikz.nodes import (
  TikzBaseNode,
  TikzCellItem,
  TikzEdge,
  TikzGraph,
  TikzNode,
  TikzOption,
  TikzTable,
  TikzTextNode,
  TriviaNode,
)

_NL = chr(10)
_BS = chr(92)


def test_base_and_trivia_nodes() -> None:
  """Verify `TikzBaseNode` default emission and `TriviaNode` whitespace/comment formatting."""

  class MinimalTikzNode(TikzBaseNode):
    """Minimal concrete subclass exercising `TikzBaseNode` default `emit()`."""

  base = MinimalTikzNode()
  assert base.emit() == ""
  assert base.to_text() == ""

  ws = TriviaNode(content="    ", kind="whitespace")
  assert ws.emit() == "    "
  assert ws.to_text() == "    "

  comment = TriviaNode(content=f"% Hello World{_NL}", kind="comment")
  assert comment.to_text() == f"% Hello World{_NL}"


def test_option_and_text_nodes() -> None:
  """Verify `TikzOption` key/value rendering and `TikzTextNode` bold/italic formatting."""
  opt_flag = TikzOption(key="draw")
  assert opt_flag.emit() == "draw"
  assert opt_flag.to_text() == "draw"

  opt_kv = TikzOption(key="fill", value="red")
  assert opt_kv.emit() == "fill=red"
  assert opt_kv.to_text() == "fill=red"

  assert TikzTextNode(content="Hello").emit() == "Hello"
  assert TikzTextNode(content="Hello", bold=True).emit() == f"{_BS}textbf{{Hello}}"
  assert TikzTextNode(content="Hello", italic=True).emit() == f"{_BS}textit{{Hello}}"
  assert TikzTextNode(content="Hello", bold=True, italic=True).emit() == f"{_BS}textbf{{{_BS}textit{{Hello}}}}"


def test_table_node_rendering() -> None:
  """Verify `TikzTable` renders alignment, mixed string/`TikzTextNode` cells, and trivia."""
  row: list[TikzCellItem] = ["plain text", TikzTextNode("bold text", bold=True)]
  table = TikzTable(
    rows=[row, ["In: 1", "Out: 32"]],
    align="c",
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode(_NL)],
  )
  text = table.emit()
  assert text.startswith(f" {_BS}begin{{tabular}}{{c}}")
  assert f"plain text & {_BS}textbf{{bold text}} {_BS}{_BS}" in text
  assert f"In: 1 & Out: 32 {_BS}{_BS}" in text
  assert text.endswith(f"{_BS}end{{tabular}}{_NL}")


def test_node_and_edge_rendering() -> None:
  """Verify `TikzNode` and `TikzEdge` emit options, coordinates, table content, connectors, and trivia."""
  node = TikzNode(
    node_id="n1",
    x=1.0,
    y=2.0,
    content="Start",
    options=[TikzOption("circle"), TikzOption("draw")],
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode(_NL)],
  )
  assert node.to_text() == f" {_BS}node [circle, draw] (n1) at (1.0, 2.0) {{Start}};{_NL}"

  table_node = TikzNode(node_id="n2", x=10, y=20, content=TikzTable(rows=[["Prop"]]))
  table_text = table_node.to_text()
  assert f"{_BS}node (n2) at (10, 20) {{" in table_text
  assert f"{_BS}begin{{tabular}}{{c}}" in table_text
  assert table_text.endswith("};")

  edge = TikzEdge(
    source_id="a",
    target_id="b",
    options=[TikzOption("->"), TikzOption("thick")],
    connector="--",
    leading_trivia=[TriviaNode(f"{_NL}    ")],
    trailing_trivia=[TriviaNode(_NL)],
  )
  assert edge.to_text() == f"{_NL}    {_BS}draw [->, thick] (a) -- (b);{_NL}"


def test_graph_composition_with_and_without_options() -> None:
  """Verify `TikzGraph` wraps child nodes, edges, and trivia in a `tikzpicture` environment."""
  node1 = TikzNode(node_id="a", x=0, y=0, content="A")
  node2 = TikzNode(node_id="b", x=1, y=0, content="B")
  edge = TikzEdge(source_id="a", target_id="b")
  graph = TikzGraph(
    options=[TikzOption("scale", "0.5")],
    children=[TriviaNode(f"{_NL}% Nodes{_NL}"), node1, node2, TriviaNode(_NL), edge, TriviaNode(_NL)],
    leading_trivia=[TriviaNode(" ")],
    trailing_trivia=[TriviaNode(_NL)],
  )
  text = graph.to_text()
  assert text.startswith(f" {_BS}begin{{tikzpicture}}[scale=0.5]")
  assert "% Nodes" in text
  assert f"{_BS}node (a) at (0, 0) {{A}};" in text
  assert f"{_BS}draw (a) -- (b);" in text
  assert text.endswith(f"{_BS}end{{tikzpicture}}{_NL}")

  plain_graph = TikzGraph(children=[node1])
  assert plain_graph.to_text() == f"{_BS}begin{{tikzpicture}}{_BS}node (a) at (0, 0) {{A}};{_BS}end{{tikzpicture}}"
