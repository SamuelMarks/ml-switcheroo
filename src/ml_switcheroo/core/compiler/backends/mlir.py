"""Docstring."""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph


class MlirBackend(CompilerBackend):
  """Back-end for generating MLIR text from a LogicalGraph.

  This implementation provides a direct Graph-to-MLIR conversion path for
  scenarios bypassing the high-level Python CST Rewriter (e.g. source is RDNA).
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize."""
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles the LogicalGraph into an MLIR string representation.

    It emits a simple `module` structure with a single block, mapping each
    logical node to an `sw.op` operation. Inputs are generated as
    `sw.constant` if metadata values exist, or `sw.op {type="Input"}`.

    Args:
        graph: The logical graph IR.

    Returns:
        str: The generated MLIR code.

    """
    lines = ["// Graph -> MLIR compilation output"]
    lines.append("module {")
    lines.append("  func.func @main() {")

    for node in graph.nodes:
      if node.kind == "Input":
        val = node.metadata.get("value", "1")
        # Try to determine type
        # If value is integer-like, cast to i32
        if str(val).isdigit():
          lines.append(f'    %{node.id} = "sw.constant"() {{value = {val}}} : () -> i32')
        else:
          # Treat input as argument or placeholder constant
          lines.append(f'    %{node.id} = "sw.op"() {{type = "Input"}} : () -> !sw.unknown')
      elif node.kind == "Output":
        # Sink node, often no output or uses return logic
        # Find source
        # Graph edges handling is simplistic here: just list nodes
        lines.append('    "sw.return"() : () -> ()')
      else:
        # Generic Op
        # Construct args string from metadata
        attrs = [f'type = "{node.kind}"']
        for k, v in node.metadata.items():
          attrs.append(f'{k} = "{v}"')
        attr_str = ", ".join(attrs)
        attr_block = f" {{{attr_str}}}" if attrs else ""
        lines.append(f'    %{node.id} = "sw.op"(){attr_block} : () -> !sw.unknown')

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
