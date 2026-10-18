"""Static Graph Extractor for TikZ Visualization.

Delegates to the canonical `GraphExtractor` in `ml_switcheroo.core.graph` while
preserving the single-input `"input"` node convention expected by TikZ visualizers.
"""

import libcst as cst

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph import GraphExtractor as _BaseGraphExtractor
from ml_switcheroo.utils.node_diff import capture_node_source

__all__ = ["GraphExtractor", "LogicalEdge", "LogicalGraph", "LogicalNode"]


class GraphExtractor(_BaseGraphExtractor):
  """LibCST Visitor that extracts a LogicalGraph from Python source code for TikZ visualization."""

  def _extract_input_args(self, node: cst.FunctionDef) -> None:
    """Register function arguments as input sources mapped to the primary `'input'` node.

    Args:
        node: The function definition CST node containing the parameters.
    """
    input_id = "input"
    self.layer_registry[input_id] = LogicalNode(input_id, "Input", attributes={})
    for param in node.params.params:
      if param.name.value == "self":
        continue
      self.provenance[param.name.value] = input_id

  def _node_to_string(self, node: cst.CSTNode) -> str:
    """Extract source string representation of an AST Node.

    Args:
        node: The CST node to convert to a string.

    Returns:
        The string representation of the source code for the node.
    """
    return capture_node_source(node)
