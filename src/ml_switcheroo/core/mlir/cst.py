"""MLIR Concrete Syntax Tree Nodes.

This module defines the data structures for representing MLIR source code.
It ensures structural hierarchy (Module -> Operation -> Region -> Block)
and trivia preservation for high-fidelity round-tripping.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from ml_switcheroo.core.cst.base import CSTNode, Trivia


@dataclass
class MlirNode(CSTNode):
  """Abstract base class for all MLIR CST nodes."""

  pass


@dataclass
class TypeNode(MlirNode):
  """Represents a type annotation."""

  body: str = ""

  def to_text(self) -> str:
    """Return textual representation."""
    out = "".join(t.text for t in self.leading_trivia)
    out += self.body
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class ValueNode(MlirNode):
  """Represents an SSA Value identifier (e.g. %0)."""

  name: str = ""
  type_node: Optional[TypeNode] = None
  colon_trivia: List[Trivia] = field(default_factory=list)

  def to_text(self) -> str:
    """Return textual representation of the node."""
    out = "".join(t.text for t in self.leading_trivia)
    out += self.name
    if self.type_node:
      out += "".join(t.text for t in self.colon_trivia)
      out += ":"
      out += self.type_node.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class AttributeNode(MlirNode):
  """Represents a named attribute."""

  name: str = ""
  value: Union[str, List[str]] = ""
  type_annotation: Optional[str] = None

  def to_text(self) -> str:
    """Return textual representation of the node."""
    out = "".join(t.text for t in self.leading_trivia)
    suffix = f" : {self.type_annotation}" if self.type_annotation else ""

    if isinstance(self.value, list):
      val_str = f"[{', '.join(self.value)}]"
    else:
      val_str = str(self.value)

    out += f"{self.name} = {val_str}{suffix}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class BlockNode(MlirNode):
  """Represents a Basic Block within a Region."""

  label: str = ""
  operations: List["OperationNode"] = field(default_factory=list)
  arguments: List[Tuple[ValueNode, TypeNode]] = field(default_factory=list)

  def to_text(self) -> str:
    """Return textual representation of the node."""
    out = "".join(t.text for t in self.leading_trivia)

    arg_strs = []
    for val, typ in self.arguments:
      arg_strs.append(f"{val.to_text()}: {typ.to_text()}")

    args_text = ""
    if arg_strs:
      args_text = f"({', '.join(arg_strs)})"

    if self.label:
      out += f"{self.label}{args_text}:"
      if not self.operations and not self.trailing_trivia:
        out += "\n"

    for op in self.operations:
      out += op.to_text()

    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class RegionNode(MlirNode):
  """Represents a Region containing Blocks."""

  blocks: List[BlockNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Return textual representation of the node."""
    out = "".join(t.text for t in self.leading_trivia)
    out += "{"
    for b in self.blocks:
      out += b.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    out += "}"
    return out


@dataclass
class OperationNode(MlirNode):
  """Represents a specific MLIR Operation."""

  name: str = ""
  results: List[ValueNode] = field(default_factory=list)
  operands: List[ValueNode] = field(default_factory=list)
  attributes: List[AttributeNode] = field(default_factory=list)
  regions: List[RegionNode] = field(default_factory=list)
  result_types: List[TypeNode] = field(default_factory=list)
  op_tail_str: str = ""
  op_tail_trivia: List[Trivia] = field(default_factory=list)
  name_trivia: List[Trivia] = field(default_factory=list)
  has_parens: bool = True

  def __post_init__(self) -> None:
    """Initialize node after dataclass creation."""
    super().__post_init__()
    if isinstance(self.name_trivia, str):
      self.name_trivia = [Trivia(self.name_trivia)]
    elif self.name_trivia is None:
      self.name_trivia = []

  def to_text(self) -> str:
    """Return textual representation of the node."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.text)

    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    parts.append(self.name)

    if self.name_trivia:
      for t in self.name_trivia:
        parts.append(t.text)

    if self.operands:
      if not self.name_trivia and not self.has_parens:
        # Check if first operand has leading trivia, if not, add space
        if not self.operands[0].leading_trivia:
          parts.append(" ")
      elif not self.name_trivia and self.has_parens:
        parts.append(" ")
      op_names = []
      for i, o in enumerate(self.operands):
        # If no trivia and not first operand, add space
        if i > 0 and not o.leading_trivia:
          op_names.append(" " + o.to_text())
        else:
          op_names.append(o.to_text())

      if self.has_parens:
        parts.append(f"({','.join(op_names)})")
      else:
        parts.append(",".join(op_names))

    if self.attributes:
      if not self.name_trivia:
        parts.append(" ")
      parts.append("{")
      attrs_str = ", ".join([a.to_text() for a in self.attributes])
      parts.append(attrs_str)
      parts.append("}")

    if self.result_types:
      for t in self.op_tail_trivia:
        parts.append(t.text)
      if not self.op_tail_str:
        if not self.op_tail_trivia and not self.name_trivia:
          parts.append(" : ")
        else:
          parts.append(": ")
      else:
        parts.append(self.op_tail_str)

      if len(self.result_types) == 1:
        parts.append(self.result_types[0].to_text())
      else:
        r_types = [t.to_text() for t in self.result_types]
        if self.op_tail_str == "-> ":
          parts.append(f"({', '.join(r_types)})")  # e.g. -> (tensor<...>, tensor<...>)
        elif not self.op_tail_str:
          parts.append(f"({', '.join(r_types)})")
        else:
          parts.append(", ".join(r_types))

    if self.regions:
      if parts and not parts[-1].endswith(" ") and not self.regions[0].leading_trivia:
        parts.append(" ")
      for reg in self.regions:
        parts.append(reg.to_text())

    for t in self.trailing_trivia:
      parts.append(t.text)

    out = "".join(parts)
    return out


@dataclass
class StableHloConstantOp(OperationNode):
  """Specialized node for stablehlo.constant preserving dialect trivia."""

  def to_text(self) -> str:
    """Return textual representation of the node."""
    parts = []
    for t in self.leading_trivia:
      parts.append(t.text)

    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    parts.append(self.name)

    if self.name_trivia:
      for t in self.name_trivia:
        parts.append(t.text)

    if self.attributes:
      if not self.name_trivia:
        parts.append(" ")
      attr = self.attributes[0]
      parts.append(str(attr.value))

    if self.result_types:
      parts.append(" : ")
      if len(self.result_types) == 1:
        parts.append(self.result_types[0].to_text())
      else:
        t_names = [t.to_text() for t in self.result_types]
        parts.append(f"({', '.join(t_names)})")

    for t in self.trailing_trivia:
      parts.append(t.text)

    out = "".join(parts)
    return out


@dataclass
class AttributeAliasDefNode(MlirNode):
  """Represents a top-level attribute alias definition."""

  name: str = ""
  value_node: Optional[MlirNode] = None
  value_str: str = ""

  def to_text(self) -> str:
    """Return textual representation."""
    out = "".join(t.text for t in self.leading_trivia)
    out += f"{self.name} = "
    if self.value_node:
      out += self.value_node.to_text()
    else:
      out += self.value_str
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class ModuleNode(MlirNode):
  """Top-level container."""

  body: BlockNode = field(default_factory=BlockNode)
  aliases: List[AttributeAliasDefNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Return textual representation of the node."""
    out = "".join(t.text for t in self.leading_trivia)
    for alias in self.aliases:
      out += alias.to_text()
    out += self.body.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    return out
