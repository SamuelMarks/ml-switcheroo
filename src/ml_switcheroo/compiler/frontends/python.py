"""
Python Frontend.

Wraps the LibCST parser and GraphExtractor to provide a standard interface
for ingesting Python code into the Logical Graph IR.
"""

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.compiler.ir import LogicalGraph


class PythonFrontend:
  """
  Ingests Python source code into a LogicalGraph.
  """

  def __init__(self, code: str) -> None:
    self.code = code

  def parse_to_graph(self) -> LogicalGraph:
    """
    Parses the code and extracts the computation graph.
    """
    try:
      tree = cst.parse_module(self.code)
    except Exception:
      # Fallback (e.g. empty code) -> Empty Graph
      return LogicalGraph()

    extractor = GraphExtractor()
    tree.visit(extractor)
    return extractor.graph
