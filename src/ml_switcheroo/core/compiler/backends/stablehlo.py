"""StableHLO Compiler Backend module.

This module provides the StableHloBackend class, which compiles a logical graph representation
of computations into StableHLO-flavored MLIR code representation.
"""

import typing

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.mlir_backend import extract_mlir_node_trivia
from ml_switcheroo.core.compiler.backends.mlir_printer import MlirPrinter
from ml_switcheroo.core.compiler.backends.python_strategies.init import _INTERNAL_TRIVIA_KEYS
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.mlir.cst import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  StableHloConstantOp,
  TypeNode,
  ValueNode,
)
from ml_switcheroo.core.mlir.types import format_mlir_tensor_type

if typing.TYPE_CHECKING:
  from ml_ecosystem_snapshots.grounding.models import GroundingReport

  from ml_switcheroo.semantics.manager import SemanticsManager

_LOGICAL_OP_ALIASES: dict[str, str] = {
  "Div": "divide",
  "Mul": "multiply",
  "Neg": "Negate",
  "Pow": "power",
  "Sub": "subtract",
  "IsFinite": "is_finite",
  "Tuple": "TUPLE",
}


class StableHloBackend(CompilerBackend):
  """Back-end for generating StableHLO text from a LogicalGraph.

  This implementation provides a direct Graph-to-StableHLO conversion path
  used when an ISA is the source format (e.g. NVIDIA_SASS -> StableHLO).
  It constructs an MLIR CST using StableHLO dialect nodes.

  Attributes:
      semantics: Optional semantic definitions or configurations to assist in translation.
  """

  def __init__(self, semantics: "SemanticsManager | None" = None) -> None:
    """Initialize the StableHloBackend.

    Args:
        semantics: Optional object containing semantic translation rules or configuration.
    """
    self.semantics = semantics

  def _resolve_stablehlo_op_name(self, op_type: str) -> str:
    """Resolve ``op_type`` to its ``stablehlo.*`` dialect operation name via ``SemanticsManager``.

    Args:
        op_type: Logical operation type string.

    Returns:
        Resolved ``stablehlo.*`` API string, or ``op_type`` if unmapped.
    """
    if not self.semantics:
      return op_type
    defn = self.semantics.get_definition(op_type)
    variants_0 = defn[1].get("variants") if defn else None
    if not defn or not isinstance(variants_0, dict) or "stablehlo" not in variants_0:
      alias = _LOGICAL_OP_ALIASES.get(op_type)
      if alias:
        defn = self.semantics.get_definition(alias)
    if defn:
      _, details = defn
      raw_variants = details.get("variants", {})
      shlo_var = raw_variants.get("stablehlo") if isinstance(raw_variants, dict) else None
      if isinstance(shlo_var, dict) and shlo_var.get("api"):
        return str(shlo_var["api"])
    return op_type

  @staticmethod
  def _build_node_attributes(attributes: typing.Mapping[str, AttributeValue]) -> list[AttributeNode]:
    """Convert non-trivia ``LogicalNode`` attributes into MLIR ``AttributeNode`` entries.

    Args:
        attributes: Raw node attribute mapping.

    Returns:
        Ordered list of ``AttributeNode`` instances.
    """
    attrs: list[AttributeNode] = []
    for k, v in attributes.items():
      if k in _INTERNAL_TRIVIA_KEYS:
        continue
      val_str = f'"{v}"' if isinstance(v, str) and not v.startswith('"') else str(v)
      attrs.append(AttributeNode(name=k, value=val_str))
    return attrs

  def _compile_node(self, node: LogicalNode, incoming_sources: list[str]) -> OperationNode:
    """Compile a single ``LogicalNode`` into its corresponding MLIR ``OperationNode``.

    Args:
        node: Logical computation node.
        incoming_sources: Ordered upstream producer node IDs.

    Returns:
        Constructed ``OperationNode`` or ``StableHloConstantOp``.
    """
    op_type = str(node.op_type)
    shape_meta = node.shape_metadata
    dtype_str = str(node.dtype or "float32")
    has_ranked_shape = isinstance(shape_meta, tuple) and bool(shape_meta)
    ranked_type_str = format_mlir_tensor_type(shape_meta, dtype_str) if has_ranked_shape else "tensor<f32>"
    leading_triv = extract_mlir_node_trivia(node.attributes)

    if op_type == "Input":
      return StableHloConstantOp(
        name="stablehlo.constant",
        results=[ValueNode(name=f"%{node.id}")],
        attributes=[AttributeNode(name="value", value="dense<0.0>")],
        result_types=[TypeNode(body=ranked_type_str)],
        leading_trivia=leading_triv,
      )
    if op_type == "Output":
      return OperationNode(
        name="return",
        operands=[ValueNode(name=f"%{src}") for src in incoming_sources],
        leading_trivia=leading_triv,
      )

    op_name = self._resolve_stablehlo_op_name(op_type)
    res_types_list = [TypeNode(body=ranked_type_str)] if has_ranked_shape else []
    attrs = self._build_node_attributes(node.attributes)

    if op_name == op_type:
      bare_name = op_type.lower().split(".")[-1]
      return OperationNode(
        name="stablehlo.custom_call",
        results=[ValueNode(name=f"%{node.id}")],
        attributes=[AttributeNode(name="call_target_name", value=f'"@{bare_name}"'), *attrs],
        result_types=res_types_list,
        leading_trivia=leading_triv,
      )
    return OperationNode(
      name=op_name,
      results=[ValueNode(name=f"%{node.id}")],
      operands=[ValueNode(name=f"%{src}") for src in incoming_sources],
      attributes=attrs,
      result_types=res_types_list,
      leading_trivia=leading_triv,
    )

  def compile(self, graph: LogicalGraph) -> str:
    """Compile the graph to StableHLO-flavored MLIR.

    Args:
        graph: The logical graph to compile.

    Returns:
        str: The generated MLIR code string using stablehlo dialect.
    """
    from collections import defaultdict

    in_edges: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
      in_edges[edge.target].append(edge.source)

    block = BlockNode(
      label="",
      operations=[self._compile_node(node, in_edges[node.id]) for node in graph.nodes.values()],
    )
    module = ModuleNode(body=block)
    printer = MlirPrinter()
    return printer.emit(module, header="// Graph -> StableHLO compilation output\n")

  def validate_op(
    self,
    op_name: str,
    operand_types: list[str] | None = None,
    attributes: dict[str, AttributeValue] | None = None,
  ) -> "GroundingReport | None":
    """Validate a StableHLO operation against grounded dialect specifications.

    Args:
        op_name: The StableHLO operation name (e.g. 'stablehlo.add').
        operand_types: Optional operand type strings.
        attributes: Operation attributes dictionary.

    Returns:
        Optional[GroundingReport]: A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.compiler import validate_stablehlo_op

      clean_op = op_name.replace("stablehlo.", "")
      report: "GroundingReport" = validate_stablehlo_op(
        op_name=clean_op,
        operand_types=operand_types or ["tensor<f32>", "tensor<f32>"],
        attributes=attributes or {},
      )
      return report
    except (ImportError, OSError, SyntaxError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
      return None
