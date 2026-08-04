"""StableHLO Compiler Backend module.

This module provides the StableHloBackend class, which compiles a logical graph representation
of computations into StableHLO-flavored MLIR code representation.
"""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.mlir.cst import (
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

  Attributes:
      semantics: Optional semantic definitions or configurations to assist in translation.
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize the StableHloBackend.

    Args:
        semantics: Optional object containing semantic translation rules or configuration.
    """
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles the graph to StableHLO-flavored MLIR.

    Args:
        graph: The logical graph to compile.

    Returns:
        str: The generated MLIR code string using stablehlo dialect.
    """
    block = BlockNode(label="")

    from collections import defaultdict

    in_edges = defaultdict(list)
    for edge in graph.edges:
      in_edges[edge.target].append(edge.source)

    for node in graph.nodes:
      op: OperationNode
      if node.kind == "Input":
        op = StableHloConstantOp(
          name="stablehlo.constant",
          results=[ValueNode(name=f"%{node.id}")],
          attributes=[AttributeNode(name="value", value="dense<0.0>")],
          result_types=[TypeNode(body="tensor<f32>")],
        )
      elif node.kind == "Output":
        operands = [ValueNode(name=f"%{src}") for src in in_edges[node.id]]
        op = OperationNode(
          name="return",
          operands=operands,
        )
      else:
        op_name = node.kind
        if self.semantics:
          defn = self.semantics.get_definition(node.kind)
          if defn:
            _, details = defn
            variants = details.get("variants", {})
            if "stablehlo" in variants and "api" in variants["stablehlo"]:
              op_name = variants["stablehlo"]["api"]

        if op_name == node.kind:
          bare_name = node.kind.lower().split(".")[-1]
          op = OperationNode(
            name="stablehlo.custom_call",
            results=[ValueNode(name=f"%{node.id}")],
            attributes=[AttributeNode(name="call_target_name", value=f'"@{bare_name}"')],
          )
        else:
          operands = [ValueNode(name=f"%{src}") for src in in_edges[node.id]]
          op = OperationNode(
            name=op_name,
            results=[ValueNode(name=f"%{node.id}")],
            operands=operands,
          )
      block.operations.append(op)

    module = ModuleNode(body=block)
    printer = MlirPrinter()
    return printer.emit(module, header="// Graph -> StableHLO compilation output\n")
