"""Docstring."""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph


class StableHloBackend(CompilerBackend):
  """Back-end for generating StableHLO text from a LogicalGraph.

  This implementation provides a direct Graph-to-StableHLO conversion path
  used when an ISA is the source format (e.g. SASS -> StableHLO).
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize."""
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles the graph to StableHLO-flavored MLIR.

    Args:
        graph: The logical graph.

    Returns:
        str: MLIR code string using stablehlo dialect.

    """
    lines = ["// Graph -> StableHLO compilation output"]
    lines.append("module {")
    lines.append("  func.func @main() {")

    for node in graph.nodes:
      if node.kind == "Input":
        lines.append(f"    %{node.id} = stablehlo.constant dense<0.0> : tensor<f32>")
      elif node.kind == "Output":
        lines.append("    return")
      else:
        # Attempt simpler mapping
        op_name = node.kind.lower().split(".")[-1]
        lines.append(f"    %{node.id} = stablehlo.custom_call @{op_name}(...)")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
