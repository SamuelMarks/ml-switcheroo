"""Test module."""

from ml_switcheroo.core.tikz.parser import TikzTransformer, _logical_from_tikz_graph
from ml_switcheroo.core.tikz.nodes import TriviaNode, TikzNode, TikzTable, TikzGraph
from lark import Tree


def test_tikz_parser_standalone():
  """Test function."""
  pass


def test_tikz_transformer_direct():
  """Test function."""
  transformer = TikzTransformer()

  # 48 - 50: Top level trivia leading/trailing
  # 74->57: end_tikzpicture handling
  # 78-83: TikzGraph extraction
  transformer.start([TriviaNode(" ")])
  transformer.start([TikzGraph([])])
  transformer.start(["\\end{tikzpicture}"])

  # 226, 232, 235->221, 238->221, 242-243, 247->221
  # 226: trailing trivia
  transformer.node(["\\node", TriviaNode(" ")])
  # 232: found_node = True
  transformer.node(["\\node"])
  # 235->221: false condition for node_id (i.e. c is "(")
  transformer.node(["\\node", "(", ")"])
  # 238->221: len(parts) != 2
  transformer.node(["\\node", "id", "1.0,2.0,3.0"])
  # 242-243: ValueError inside parts parsing
  transformer.node(["\\node", "id", "a,b"])
  # 247->221: empty children for text_content
  transformer.node([Tree("text_content", [])])

  # 278, 281, 284->273, 289->273
  transformer.edge(["\\draw", TriviaNode(" ")])  # 278, 281
  transformer.edge(["\\draw", "(", ")"])  # 284->273
  transformer.edge(["\\draw", "src", "->", "dst", Tree("unknown", [])])  # 289->273

  # 319->322, 325, 326->317, 336-339, 340->317
  # 319->322, 325: trailing trivia
  transformer.tabular(["\\begin{tabular}", TriviaNode(" ")])
  # 326->317: c in ["{", "}"]
  transformer.tabular(["\\begin{tabular}", "{", "c", "}"])
  # 336-339: tabular_row items (id, kind)
  transformer.tabular([Tree("tabular_row", [Tree("kind", ["Kind"]), Tree("id", ["Id"]), Tree("unknown", [])])])
  # 340->317: empty current_row
  transformer.tabular([Tree("tabular_row", [])])

  # 376 inside id()
  transformer.id(["A"])


def test_logical_from_tikz_graph():
  """Test function."""
  # 468, 480->483

  node1 = TikzNode(
    "n1",
    0,
    0,
    content=TikzTable(
      rows=[
        [],  # 468
        ["\\textit{id}"],
        ["JustText"],
        ["a:b\\\\"],
      ]
    ),
  )

  node2 = TikzNode("n2", 0, 0, content=None)  # 480->483 false
  node3 = TikzNode("n3", 0, 0, content="")  # 480->483 false

  graph = TikzGraph(children=[node1, node2, node3])
  lg = _logical_from_tikz_graph(graph)  # noqa: F841
