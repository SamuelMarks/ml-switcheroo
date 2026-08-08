"""Python Frontend.

Wraps the LibCST parser and GraphExtractor to provide a standard interface
for ingesting Python code into the Logical Graph IR.
"""

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.compiler.ir import LogicalGraph


class PythonFrontend:
  """Ingests Python source code into a LogicalGraph.

  This class acts as the frontend interface for converting Python source code
  into a Logical Graph intermediate representation (IR) by utilizing a LibCST
  parser and a custom GraphExtractor visitor.
  """

  def __init__(self, code: str) -> None:
    """Initializes the PythonFrontend with the target source code.

    Args:
        code: The Python source code string to be parsed and analyzed.
    """
    self.code = code

  def parse_to_graph(self) -> LogicalGraph:
    """Parses the Python source code and extracts the computation graph.

    This method parses the stored Python code into a concrete syntax tree (CST)
    and uses the GraphExtractor to visit the tree nodes, building up the
    LogicalGraph. In case of syntax or parsing errors (e.g., empty code),
    it safely falls back and returns an empty LogicalGraph.

    Returns:
        LogicalGraph: A LogicalGraph representing the extracted computation graph structure.
    """
    try:
      tree = cst.parse_module(self.code)
    except Exception:
      # Fallback (e.g. empty code) -> Empty Graph
      return LogicalGraph()

    extractor = GraphExtractor()
    tree.visit(extractor)
    return extractor.graph
