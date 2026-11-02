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
from ml_switcheroo.core.compiler.ir import LogicalGraph
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

  def compile(self, graph: LogicalGraph) -> str:
    """Compile the graph to StableHLO-flavored MLIR.

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

    for node in graph.nodes.values():
      op: OperationNode
      op_type: str = str(getattr(node, "op_type", None) or getattr(node, "kind", ""))
      node_attrs = getattr(node, "attributes", {})
      shape_meta = getattr(node, "shape_metadata", None)
      dtype_str = str(getattr(node, "dtype", None) or "float32")
      has_ranked_shape = isinstance(shape_meta, tuple) and bool(shape_meta)
      ranked_type_str = format_mlir_tensor_type(shape_meta, dtype_str) if has_ranked_shape else "tensor<f32>"
      leading_triv = extract_mlir_node_trivia(node_attrs)

      if op_type == "Input":
        op = StableHloConstantOp(
          name="stablehlo.constant",
          results=[ValueNode(name=f"%{node.id}")],
          attributes=[AttributeNode(name="value", value="dense<0.0>")],
          result_types=[TypeNode(body=ranked_type_str)],
          leading_trivia=leading_triv,
        )
      elif op_type == "Output":
        operands = [ValueNode(name=f"%{src}") for src in in_edges[node.id]]
        op = OperationNode(
          name="return",
          operands=operands,
          leading_trivia=leading_triv,
        )
      else:
        op_name = op_type
        if self.semantics:
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
              op_name = str(shlo_var["api"])

        res_types_list = [TypeNode(body=ranked_type_str)] if has_ranked_shape else []
        attrs = []
        for k, v in node.attributes.items():
          if k in _INTERNAL_TRIVIA_KEYS:
            continue
          if isinstance(v, str) and not v.startswith('"'):
            attrs.append(AttributeNode(name=k, value=f'"{v}"'))
          else:
            attrs.append(AttributeNode(name=k, value=str(v)))

        if op_name == op_type:
          bare_name = op_type.lower().split(".")[-1]
          op = OperationNode(
            name="stablehlo.custom_call",
            results=[ValueNode(name=f"%{node.id}")],
            attributes=[AttributeNode(name="call_target_name", value=f'"@{bare_name}"'), *attrs],
            result_types=res_types_list,
            leading_trivia=leading_triv,
          )
        else:
          operands = [ValueNode(name=f"%{src}") for src in in_edges[node.id]]
          op = OperationNode(
            name=op_name,
            results=[ValueNode(name=f"%{node.id}")],
            operands=operands,
            attributes=attrs,
            result_types=res_types_list,
            leading_trivia=leading_triv,
          )
      block.operations.append(op)

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
    except Exception:
      return None
