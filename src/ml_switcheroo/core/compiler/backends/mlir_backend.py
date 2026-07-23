"""MLIR Compiler Backend."""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  AttributeNode,
  ValueNode,
  TypeNode,
)
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter


class MlirBackend(CompilerBackend):
  """Back-end for generating MLIR text from a LogicalGraph.

  This implementation constructs an MLIR CST (Concrete Syntax Tree)
  mapping each logical node to an `sw.op` operation, and then prints it.
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
    block = BlockNode(label="")

    for node in graph.nodes:
      if node.kind == "Input":
        val = node.metadata.get("value", "1")
        # Try to determine type
        if str(val).isdigit():
          op = OperationNode(
            name='"sw.constant"',
            results=[ValueNode(f"%{node.id}")],
            attributes=[AttributeNode(name="value", value=str(val))],
            result_types=[TypeNode("i32")],
          )
        else:
          op = OperationNode(  # pragma: no cover
            name='"sw.op"',
            results=[ValueNode(f"%{node.id}")],
            attributes=[AttributeNode(name="type", value='"Input"')],
            result_types=[TypeNode("!sw.unknown")],
          )
      elif node.kind == "Output":
        op = OperationNode(name='"sw.return"', result_types=[TypeNode("()")])
      else:
        # Generic Op
        attrs = [AttributeNode(name="type", value=f'"{node.kind}"')]
        for k, v in node.metadata.items():
          attrs.append(AttributeNode(name=k, value=f'"{v}"'))

        op = OperationNode(
          name='"sw.op"', results=[ValueNode(f"%{node.id}")], attributes=attrs, result_types=[TypeNode("!sw.unknown")]
        )
      block.operations.append(op)

    module = ModuleNode(body=block)
    printer = MlirPrinter()
    return printer.emit(module)
