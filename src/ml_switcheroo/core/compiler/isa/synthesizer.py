"""Unified ISA Synthesizer.

Provides the shared middle-end logic for hardware ISA synthesizers (`nvidia_sass`, `rdna`),
including macro registry loading, liveness-tracked `from_graph` lowering, and `to_python`
LibCST lifting.
"""

import json
import os
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import libcst as cst
import yaml

from ml_switcheroo.core.compiler.ir import LogicalGraph, topological_sort
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator


class IsaSynthesizer:
  """Unified Base Class for Hardware ISA Synthesizers."""

  TARGET_NAME: str = ""
  COMMENT_CLS: Type[Any] = object
  INSTRUCTION_CLS: Type[Any] = object
  LABEL_CLS: Type[Any] = object
  IMMEDIATE_CLS: Type[Any] = object
  LABEL_COMMENT_PREFIX: str = "Label"
  EMIT_BEGIN_END_COMMENTS_IN_PY: bool = False

  def __init__(
    self,
    semantics: Any,
    allocator: GenericRegisterAllocator[Any],
    macros_yaml_path: str,
    macros_json_path: str,
    macro_module: Any,
  ) -> None:
    """Initialize the ISA synthesizer with semantics, register allocator, and macro mappings.

    Args:
        semantics: SemanticsManager instance for resolving abstract operations.
        allocator: Register allocator managing physical register assignments.
        macros_yaml_path: Filesystem path to the YAML macro registry file.
        macros_json_path: Fallback filesystem path to the JSON macro registry file.
        macro_module: Python module containing macro expansion functions.
    """
    self.semantics = semantics
    self.allocator = allocator

    self.macro_registry: Dict[str, Callable[..., List[Any]]] = {}
    mapping: Dict[str, Any] = {}
    if os.path.exists(macros_yaml_path):
      try:
        with open(macros_yaml_path, "r", encoding="utf-8") as f:
          mapping = yaml.safe_load(f) or {}
      except (OSError, yaml.YAMLError):
        mapping = {}

    if not mapping and os.path.exists(macros_json_path):
      try:
        with open(macros_json_path, "r", encoding="utf-8") as f:
          mapping = json.load(f)
      except (OSError, json.JSONDecodeError):
        mapping = {}

    for key, func_name in mapping.items():
      if hasattr(macro_module, func_name):
        self.macro_registry[key] = getattr(macro_module, func_name)

  def _resolve_abstract_id(self, op_type: str) -> str:
    """Resolve an operation type string to its canonical abstract operation ID.

    Args:
        op_type: Operation type or qualified API name.

    Returns:
        Resolved abstract operation ID, or ``op_type`` if no definition is found.
    """
    defn = self.semantics.get_definition(op_type)
    if not defn and ("." in op_type):
      suffix = op_type.split(".", 1)[-1]
      defn = self.semantics.get_definition(suffix)
    return defn[0] if defn else op_type

  def _get_input_map(self, graph: LogicalGraph) -> Dict[str, List[str]]:
    """Build an input adjacency map from target node IDs to ordered source node IDs.

    Args:
        graph: The LogicalGraph to inspect.

    Returns:
        Dictionary mapping each target node ID to a list of incoming source node IDs.
    """
    input_map: Dict[str, List[str]] = {}
    for edge in graph.edges:
      if edge.target not in input_map:
        input_map[edge.target] = []
      input_map[edge.target].append(edge.source)
    return input_map

  def _format_reg_for_comment(self, reg: Any) -> str:
    """Format a physical register for input/return semantic comments.

    Args:
        reg: Allocated register object.

    Returns:
        String representation of the register.
    """
    return str(getattr(reg, "name", reg))

  def _handle_custom_opcode(
    self,
    opcode: str,
    node_id: str,
    node_attrs: Dict[str, Any],
  ) -> Optional[List[Any]]:
    """Handle target-specific macro directives or comment opcodes before ALU emission.

    Args:
        opcode: Resolved target opcode string.
        node_id: Logical node identifier.
        node_attrs: Logical node attributes dictionary.

    Returns:
        List of emitted ISA AST nodes if handled, or ``None`` to proceed with standard ALU emission.
    """
    return None

  def from_graph(self, graph: LogicalGraph) -> List[Any]:
    """Synthesize a linear sequence of ISA AST nodes from a `LogicalGraph`.

    Args:
        graph: LogicalGraph intermediate representation to lower.

    Returns:
        Ordered list of synthesized ISA instructions, labels, and comments.
    """
    self.allocator.reset()
    self.allocator.build_liveness(graph)
    output_nodes: List[Any] = []
    sorted_nodes = topological_sort(graph)
    input_map = self._get_input_map(graph)

    for node in sorted_nodes:
      op_type: str = str(node.op_type if hasattr(node, "op_type") else getattr(node, "kind", ""))
      node_attrs = getattr(node, "attributes", {})

      if op_type == "Input":
        reg = self.allocator.get_register(node.id)
        var_name = node_attrs.get("name", node.id)
        output_nodes.append(self.COMMENT_CLS(text=f"Input {var_name} -> {self._format_reg_for_comment(reg)}"))

      elif op_type == "Output":
        sources = input_map.get(node.id, [])
        if sources:
          src_reg = self.allocator.get_register(sources[0])
          output_nodes.append(self.COMMENT_CLS(text=f"Return: {self._format_reg_for_comment(src_reg)}"))
          self.allocator.record_usage(sources[0])

      else:
        abstract_id = self._resolve_abstract_id(op_type)

        if abstract_id in self.macro_registry:
          expander = self.macro_registry[abstract_id]
          output_nodes.extend(expander(self.allocator, node.id, node_attrs))
          for src_id in input_map.get(node.id, []):
            self.allocator.record_usage(src_id)
          continue

        suffix_id = abstract_id.split(".")[-1] if abstract_id else ""
        if suffix_id and suffix_id in self.macro_registry:
          expander = self.macro_registry[suffix_id]
          output_nodes.extend(expander(self.allocator, node.id, node_attrs))
          for src_id in input_map.get(node.id, []):
            self.allocator.record_usage(src_id)
          continue

        variant = self.semantics.resolve_variant(abstract_id, self.TARGET_NAME) if abstract_id else None
        if not variant or not variant.get("api"):
          output_nodes.append(self.COMMENT_CLS(text=f"Unmapped Op: {op_type} ({node.id})"))
          continue

        opcode = variant["api"]
        custom_nodes = self._handle_custom_opcode(opcode, node.id, node_attrs)
        if custom_nodes is not None:
          output_nodes.extend(custom_nodes)
          continue

        dst_reg = self.allocator.get_register(node.id)
        operands: List[Any] = [dst_reg]
        for src_id in input_map.get(node.id, []):
          src_reg = self.allocator.get_register(src_id)
          operands.append(src_reg)
          self.allocator.record_usage(src_id)

        output_nodes.append(self.INSTRUCTION_CLS(opcode=opcode, operands=operands))

    return output_nodes

  def to_python(self, isa_nodes: Sequence[Any]) -> cst.Module:
    """Convert a sequence of ISA AST nodes into a Python LibCST Module.

    Args:
        isa_nodes: Sequence of ISA AST nodes to lift into Python statements.

    Returns:
        Constructed LibCST Module representing the instruction stream.
    """
    body_stmts: List[cst.SimpleStatementLine] = []
    for node in isa_nodes:
      stmt: Optional[cst.SimpleStatementLine] = None
      if isinstance(node, self.INSTRUCTION_CLS):
        stmt = self._convert_instruction_to_py(node)
      elif self.EMIT_BEGIN_END_COMMENTS_IN_PY and isinstance(node, self.COMMENT_CLS):
        text = getattr(node, "text", "")
        if "BEGIN" in text or "END" in text:
          stmt = cst.SimpleStatementLine(
            body=[cst.Pass()],
            trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(value=f"# {text}")),
          )
      elif isinstance(node, self.LABEL_CLS):
        label_name = getattr(node, "name", "")
        stmt = cst.SimpleStatementLine(
          body=[cst.Pass()],
          trailing_whitespace=cst.TrailingWhitespace(
            comment=cst.Comment(value=f"# {self.LABEL_COMMENT_PREFIX}: {label_name}")
          ),
        )

      if stmt is not None:
        body_stmts.append(stmt)
    return cst.Module(body=body_stmts)

  @abstractmethod
  def _convert_instruction_to_py(self, inst: Any) -> cst.SimpleStatementLine:
    """Convert a single ISA instruction node into a Python statement.

    Args:
        inst: The ISA instruction CST node to convert.

    Returns:
        LibCST SimpleStatementLine invoking `<target>.<opcode>(...)`.
    """

  def _convert_operand_to_py(self, op: Any) -> cst.BaseExpression:
    """Convert an ISA operand node into a Python LibCST expression.

    Args:
        op: The ISA operand node to convert.

    Returns:
        Corresponding LibCST Integer, Float, Name, or SimpleString expression.
    """
    if isinstance(op, self.IMMEDIATE_CLS):
      if getattr(op, "is_hex", False):
        return cst.Integer(hex(int(op.value)))
      if isinstance(op.value, float):
        return cst.Float(str(op.value))
      return cst.Integer(str(int(op.value)))

    raw = str(op)
    if "[" in raw:
      clean = raw.replace("[", "_").replace("]", "").replace(":", "_")
      return cst.Name(clean)

    if (raw.isalnum() or "_" in raw) and not raw.isdigit():
      return cst.Name(raw)
    return cst.SimpleString(f"'{raw}'")

  def _make_call(self, opcode: str, args: List[cst.Arg]) -> cst.Call:
    """Construct a `<TARGET_NAME>.<opcode>(*args)` LibCST Call node.

    Args:
        opcode: ISA instruction mnemonic.
        args: List of LibCST Arg nodes for the instruction operands.

    Returns:
        Constructed LibCST Call node.
    """
    return cst.Call(func=cst.Attribute(value=cst.Name(self.TARGET_NAME), attr=cst.Name(opcode)), args=args)
