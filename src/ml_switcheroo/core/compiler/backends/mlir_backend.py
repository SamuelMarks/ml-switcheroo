"""MLIR Compiler Backend."""

import typing

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.backends.python_strategies.init import _INTERNAL_TRIVIA_KEYS
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.types import format_mlir_tensor_type

if typing.TYPE_CHECKING:
  from ml_ecosystem_snapshots.grounding.models import GroundingReport

  from ml_switcheroo.semantics.manager import SemanticsManager


def extract_mlir_node_trivia(node_attrs: typing.Mapping[str, AttributeValue]) -> list[Trivia]:
  """Convert ``leading_trivia``, ``trailing_trivia``, and ``runtime_comment`` into MLIR ``Trivia`` nodes.

  Args:
      node_attrs: ``LogicalNode.attributes`` mapping.

  Returns:
      List of ``Trivia`` items formatted as ``// ...``.
  """
  out: list[Trivia] = []
  for key in ("leading_trivia", "trailing_trivia", "runtime_comment"):
    raw = node_attrs.get(key)
    lines = (
      [s.strip() for s in raw.splitlines() if s.strip()]
      if isinstance(raw, str)
      else ([str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else [])
    )
    for line in lines:
      clean = line.lstrip("#;/ ").strip()
      out.append(Trivia(f"// {clean}"))
      out.append(Trivia("\n"))
  return out


class MlirBackend(CompilerBackend):
  """Back-end for generating MLIR text from a LogicalGraph.

  This implementation constructs an MLIR CST (Concrete Syntax Tree)
  mapping each logical node to an `sw.op` operation, and then prints it.
  """

  def __init__(self, semantics: "SemanticsManager | None" = None) -> None:
    """Initialize the MLIR backend.

    Args:
        semantics: Optional semantics manager to configure generation behaviors.
    """
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """Compile the LogicalGraph into an MLIR string representation.

    It emits a simple `module` structure with a single block, mapping each
    logical node to an `sw.op` operation. Inputs are generated as
    `sw.constant` if metadata values exist, or `sw.op {type="Input"}`.

    Args:
        graph: The logical graph IR.

    Returns:
        str: The generated MLIR code.

    """
    block = BlockNode(label="")

    for node in graph.nodes.values():
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      node_attrs = getattr(node, "attributes", {})
      shape_meta = getattr(node, "shape_metadata", None)
      dtype_str = str(getattr(node, "dtype", None) or "float32")
      res_type_body = (
        format_mlir_tensor_type(shape_meta, dtype_str) if isinstance(shape_meta, tuple) and shape_meta else "!sw.unknown"
      )
      leading_triv = extract_mlir_node_trivia(node_attrs)

      if op_type == "Input":
        val = node_attrs.get("value", "1")
        # Try to determine type
        if str(val).isdigit() and not (isinstance(shape_meta, tuple) and shape_meta):
          op = OperationNode(
            name='"sw.constant"',
            results=[ValueNode(name=f"%{node.id}")],
            attributes=[AttributeNode(name="value", value=str(val))],
            result_types=[TypeNode(body="i32")],
            leading_trivia=leading_triv,
          )
        else:
          op = OperationNode(
            name='"sw.op"',
            results=[ValueNode(name=f"%{node.id}")],
            attributes=[AttributeNode(name="type", value='"Input"')],
            result_types=[TypeNode(body=res_type_body)],
            leading_trivia=leading_triv,
          )
      elif op_type == "Output":
        op = OperationNode(
          name='"sw.return"',
          result_types=[TypeNode(body="()")],
          leading_trivia=leading_triv,
        )
      else:
        # Generic Op
        attrs = [AttributeNode(name="type", value=f'"{op_type}"')]
        for k, v in node_attrs.items():
          if k in _INTERNAL_TRIVIA_KEYS:
            continue
          attrs.append(AttributeNode(name=k, value=f'"{v}"'))

        op = OperationNode(
          name='"sw.op"',
          results=[ValueNode(name=f"%{node.id}")],
          attributes=attrs,
          result_types=[TypeNode(body=res_type_body)],
          leading_trivia=leading_triv,
        )
      block.operations.append(op)

    module = ModuleNode(body=block)
    printer = MlirPrinter()
    return printer.emit(module)

  def validate_op(
    self,
    op_name: str,
    operand_count: int = 0,
    attributes: dict[str, AttributeValue] | None = None,
  ) -> "GroundingReport | None":
    """Validate an MLIR operation against grounded dialect specifications.

    Args:
        op_name: The MLIR operation name (e.g. 'func.func', 'arith.addi').
        operand_count: Number of input operands.
        attributes: Operation attributes dictionary.

    Returns:
        Optional[GroundingReport]: A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.compiler import validate_mlir_op

      if "." in op_name:
        dialect, name = op_name.split(".", 1)
      else:
        dialect, name = "builtin", op_name
      report: "GroundingReport" = validate_mlir_op(
        dialect=dialect,
        op_name=name,
        operand_count=operand_count,
        attributes=attributes or {},
      )
      return report
    except Exception:
      return None
