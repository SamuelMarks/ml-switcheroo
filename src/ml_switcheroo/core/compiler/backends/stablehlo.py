"""StableHLO Compiler Backend."""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  StableHloConstantOp,
  AttributeNode,
  ValueNode,
  TypeNode,
)
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter


class StableHloBackend(CompilerBackend):
  """Back-end for generating StableHLO text from a LogicalGraph.

  This implementation provides a direct Graph-to-StableHLO conversion path
  used when an ISA is the source format (e.g. SASS -> StableHLO).
  It constructs an MLIR CST using StableHLO dialect nodes.
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
    block = BlockNode(label="")

    for node in graph.nodes:
      op: OperationNode
      if node.kind == "Input":
        op = StableHloConstantOp(
          name="stablehlo.constant",
          results=[ValueNode(f"%{node.id}")],
          attributes=[AttributeNode(name="value", value="dense<0.0>")],
          result_types=[TypeNode("tensor<f32>")],
        )
      elif node.kind == "Output":
        op = OperationNode(
          name="return",
        )
      else:
        # Attempt simpler mapping
        op_name = node.kind.lower().split(".")[-1]
        op = OperationNode(
          name="stablehlo.custom_call",
          results=[ValueNode(f"%{node.id}")],
          attributes=[AttributeNode(name="call_target_name", value=f'"@{op_name}"')],
        )
      block.operations.append(op)

    module = ModuleNode(body=block)
    printer = MlirPrinter()
    return printer.emit(module, header="// Graph -> StableHLO compilation output\n")
