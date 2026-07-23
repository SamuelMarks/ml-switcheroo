"""TikZ Parser (CST Reconstruction).

This module provides the `TikzParser` which consumes raw LaTeX/TikZ source code
and reconstructs the `TikzGraph` representation using a formal Lark grammar.
"""

import os
from typing import List, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
  from ml_switcheroo.core.graph import LogicalGraph

from lark import Lark, Transformer, Tree, Token
from ml_switcheroo.core.tikz.nodes import (
  TikzBaseNode,
  TikzGraph,
  TikzNode,
  TikzEdge,
  TikzTable,
  TikzOption,
  TriviaNode,
)


class TikzTransformer(Transformer[Token, Any]):
  """Transforms a parsed TikZ AST into TikZ CST components."""

  def start(self, children: List[Any]) -> TikzGraph:
    """Process the root start rule."""
    elements = []
    leading = []
    trailing = []
    for c in children:
      if isinstance(c, list):
        # elements
        elements.extend(c)
      elif isinstance(c, TriviaNode):  # pragma: no cover
        # Handle top level trivia if not absorbed by elements
        if not elements:  # pragma: no cover
          leading.append(c)  # pragma: no cover
        else:
          trailing.append(c)  # pragma: no cover

    # Extract Graph from elements
    graph_children: List[TikzBaseNode] = []
    found_env = False
    options: List[TikzOption] = []
    for el in elements:
      if isinstance(el, TriviaNode):
        if not found_env:
          leading.append(el)
        else:
          graph_children.append(el)
      elif isinstance(el, str):
        if el.startswith("\\begin{tikzpicture}"):
          found_env = True
          # extract options if any
          if "[" in el and "]" in el:
            opts_str = el.split("[")[1].split("]")[0]
            opts = [o.strip() for o in opts_str.split(",")]
            for o in opts:
              options.append(TikzOption(key=o))
      elif isinstance(el, (TikzNode, TikzEdge)):
        graph_children.append(el)
      elif isinstance(el, TikzGraph):  # pragma: no cover
        # If the parser successfully mapped the whole thing to a graph
        # we just return it with attached trivia
        el.leading_trivia = leading + el.leading_trivia  # pragma: no cover
        el.trailing_trivia = el.trailing_trivia + trailing  # pragma: no cover
        return el  # pragma: no cover

    # Fallback if no TIKZ_ENV_BEGIN was matched properly
    return TikzGraph(children=graph_children, options=options, leading_trivia=leading, trailing_trivia=trailing)

  def element(self, children: List[Any]) -> List[Any]:
    """Process an element."""
    return children

  def TIKZ_ENV_BEGIN(self, token: Token) -> str:
    """Process a TIKZ_ENV_BEGIN token."""
    return str(token)

  def NAME(self, token: Token) -> str:
    """Process a NAME token."""
    return str(token)

  def OPTION(self, token: Token) -> TikzOption:
    """Process an OPTION token."""
    return TikzOption(key=str(token))

  def COORD(self, token: Token) -> str:
    """Process a COORD token."""
    return str(token)

  def ALIGN(self, token: Token) -> str:
    """Process an ALIGN token."""
    return str(token)

  def EDGE_OP(self, token: Token) -> str:
    """Process an EDGE_OP token."""
    return str(token)

  def trivia(self, tokens: List[Token]) -> TriviaNode:
    """Combine WS and COMMENT tokens into a TriviaNode."""
    content = "".join(str(t) for t in tokens)
    return TriviaNode(content=content)

  def IGNORE_TEXT(self, token: Token) -> TriviaNode:
    """Process an IGNORE_TEXT token."""
    return TriviaNode(content=str(token))  # pragma: no cover

  def node(self, children: List[Any]) -> TikzNode:
    """Process a node declaration."""
    leading = []
    node_id = ""
    x = 0.0
    y = 0.0
    options = []
    content: Union[str, TikzTable] = ""
    trailing = []

    # State tracking
    found_node = False

    for c in children:
      if isinstance(c, TriviaNode):
        if not found_node:
          leading.append(c)
        else:
          trailing.append(c)  # pragma: no cover
      elif isinstance(c, TikzOption):
        options.append(c)
      elif isinstance(c, str):
        # It's a token like \node or structural string from grammar, or a wrapped token
        if c == "\\node":
          found_node = True  # pragma: no cover
        elif not node_id and c not in ["(", ")", "at", "[", "]", "{", "}", ";", "\\node"]:
          node_id = c
        elif c not in ["(", ")", "at", "[", "]", "{", "}", ";", "\\node"]:
          # COORD could be string
          parts = str(c).split(",")
          if len(parts) == 2:
            try:
              x = float(parts[0].strip())
              y = float(parts[1].strip())
            except ValueError:  # pragma: no cover
              pass  # pragma: no cover
      elif isinstance(c, TikzTable):
        content = c
      elif isinstance(c, Tree) and c.data == "text_content":
        if c.children:
          content = str(c.children[0])

    return TikzNode(
      node_id=node_id, x=x, y=y, content=content, options=options, leading_trivia=leading, trailing_trivia=trailing
    )

  def edge(self, children: List[Any]) -> TikzEdge:
    """Process an edge declaration."""
    leading = []
    source_id = ""
    target_id = ""
    options = []
    connector = "--"
    trailing = []

    found_draw = False

    for c in children:
      if isinstance(c, TriviaNode):
        if not found_draw:
          leading.append(c)
        else:
          trailing.append(c)  # pragma: no cover
      elif isinstance(c, str):
        if c == "\\draw":
          found_draw = True  # pragma: no cover
        elif c in ["--", "->"]:
          connector = c
        elif c not in ["(", ")", "[", "]", ";", "\\draw"]:
          if not source_id:
            source_id = c
          else:
            target_id = c
      elif isinstance(c, TikzOption):
        options.append(c)

    return TikzEdge(
      source_id=source_id,
      target_id=target_id,
      options=options,
      connector=connector,
      leading_trivia=leading,
      trailing_trivia=trailing,
    )

  def tabular(self, children: List[Any]) -> TikzTable:
    """Process a tabular."""
    align = "c"
    rows: List[List[str]] = []
    leading: List[TriviaNode] = []
    trailing: List[TriviaNode] = []

    found_begin = False

    for c in children:
      if isinstance(c, TriviaNode):
        if not found_begin:
          leading.append(c)
        else:
          pass
      elif isinstance(c, str):
        if c == "\\begin{tabular}":
          found_begin = True  # pragma: no cover
        elif c not in ["{", "}", "\\end{tabular}"]:
          align = c
      elif isinstance(c, Tree) and c.data == "tabular_row":
        current_row = []
        for item in c.children:
          if isinstance(item, Tree):
            if item.data == "ignore":
              current_row.append(str(item.children[0]).replace("\\\\", "").strip())
            elif item.data == "meta":
              current_row.append(f"{item.children[0]}: {item.children[2]}")
            elif item.data == "kind":  # pragma: no cover
              current_row.append(f"\\textbf{{{item.children[0]}}}")  # pragma: no cover
            elif item.data == "id":  # pragma: no cover
              current_row.append(f"\\textit{{{item.children[0]}}}")  # pragma: no cover
        if current_row:
          rows.append(current_row)

    return TikzTable(align=align, rows=rows, leading_trivia=leading, trailing_trivia=trailing)

  def tabular_row(self, children: List[Any]) -> Tree[Token]:
    """Process a tabular row."""
    return Tree("tabular_row", children)

  def kind(self, children: List[Any]) -> Tree[Token]:
    """Process a kind node."""
    return Tree("kind", children)  # pragma: no cover

  def id(self, children: List[Any]) -> Tree[Token]:
    """Process an id node."""
    return Tree("id", children)  # pragma: no cover

  def meta(self, children: List[Any]) -> Tree[Token]:
    """Process a meta node."""
    return Tree("meta", children)

  def ignore(self, children: List[Any]) -> Tree[Token]:
    """Process an ignore node."""
    return Tree("ignore", children)


class TikzParser:
  """Parses TikZ code into a TikzGraph (CST) using a formal Lark grammar."""

  def __init__(self, text: str) -> None:
    """Initialize parser and tokenize input."""
    self.text = text

    grammar_path = os.path.join(os.path.dirname(__file__), "grammar.lark")
    with open(grammar_path, "r", encoding="utf-8") as f:
      self.grammar = f.read()

    self.parser = Lark(self.grammar, start="start", parser="earley")

  def parse(self) -> TikzGraph:
    """Parse the input TikZ text and construct the graph.

    Returns:
        The reconstructed TikzGraph.
    """
    try:
      tree = self.parser.parse(self.text)
    except Exception as e:
      # Re-raise standard exception for fuzzer/error handling tests
      raise ValueError(f"Failed to parse TikZ: {e}") from e

    transformer = TikzTransformer()
    from typing import cast

    return cast(TikzGraph, transformer.transform(tree))


def _logical_from_tikz_graph(tikz_graph: TikzGraph) -> "LogicalGraph":
  """Adapter to convert TikzGraph (CST) to LogicalGraph (IR)."""
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  l_graph = LogicalGraph()
  for child in tikz_graph.children:
    if isinstance(child, TikzNode):
      kind = "Unknown"
      metadata = {}
      if isinstance(child.content, TikzTable):
        for row in child.content.rows:
          if not row:
            continue  # pragma: no cover
          item = row[0]
          if "\\textbf{" in item:
            kind = item.split("\\textbf{")[1].split("}")[0].strip()
          elif "\\textit{" in item:
            pass  # id
          elif ":" in item:
            k, v = item.split(":", 1)
            k = k.replace("\\_", "_").strip()
            v = v.replace("\\_", "_").replace("\\\\", "").strip()
            metadata[k] = v
      elif isinstance(child.content, str) and child.content:
        kind = child.content.strip()

      l_graph.nodes.append(LogicalNode(id=child.node_id, kind=kind, metadata=metadata))
    elif isinstance(child, TikzEdge):
      l_graph.edges.append(LogicalEdge(source=child.source_id, target=child.target_id))

  return l_graph
