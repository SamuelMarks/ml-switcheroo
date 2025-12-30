"""
MLIR Concrete Syntax Tree Nodes.

This module defines the data structures for representing MLIR source code.
It ensures structural hierarchy (Module -> Operation -> Region -> Block)
and trivia preservation for high-fidelity round-tripping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class MlirNode(ABC):
  """Abstract base class for all MLIR CST nodes."""

  @abstractmethod
  def to_text(self) -> str:
    pass


@dataclass
class TriviaNode(MlirNode):
  """Represents non-semantic tokens (whitespace, comments)."""

  content: str
  kind: str = "whitespace"

  def to_text(self) -> str:
    return self.content


@dataclass
class ValueNode(MlirNode):
  """Represents an SSA Value identifier (e.g. %0)."""

  name: str

  def to_text(self) -> str:
    return self.name


@dataclass
class TypeNode(MlirNode):
  """Represents a type annotation."""

  body: str

  def to_text(self) -> str:
    return self.body


@dataclass
class AttributeNode(MlirNode):
  """
  Represents a named attribute.
  Value can be a string literal or a list of string literals (e.g. for bases).
  """

  name: str
  value: Union[str, List[str]]
  type_annotation: Optional[str] = None

  def to_text(self) -> str:
    suffix = f" : {self.type_annotation}" if self.type_annotation else ""

    if isinstance(self.value, list):
      # Format as MLIR array: ["val1", "val2"]
      val_str = f"[{', '.join(self.value)}]"
    else:
      val_str = self.value

    return f"{self.name} = {val_str}{suffix}"


@dataclass
class BlockNode(MlirNode):
  """Represents a Basic Block within a Region."""

  label: str
  operations: List["OperationNode"] = field(default_factory=list)
  arguments: List[Tuple[ValueNode, TypeNode]] = field(default_factory=list)
  leading_trivia: List[TriviaNode] = field(default_factory=list)

  def to_text(self) -> str:
    out = []
    for t in self.leading_trivia:
      out.append(t.to_text())

    arg_strs = []
    for val, typ in self.arguments:
      arg_strs.append(f"{val.to_text()}: {typ.to_text()}")

    args_text = ""
    if arg_strs:
      args_text = f"({', '.join(arg_strs)})"

    if self.label:
      out.append(f"{self.label}{args_text}:")
      # Note: We rely on the first operation's leading trivia to provide the newline
      # to prevent double-newlines during round-tripping.
      if not self.operations:
        # Edge case: Empty block needs newline if valid label exists
        out.append("\n")

    for op in self.operations:
      out.append(op.to_text())

    return "".join(out)


@dataclass
class RegionNode(MlirNode):
  """Represents a Region containing Blocks."""

  blocks: List[BlockNode] = field(default_factory=list)

  def to_text(self) -> str:
    blocks_text = [b.to_text() for b in self.blocks]
    content = "".join(blocks_text)
    return "{" + content + "}"


@dataclass
class OperationNode(MlirNode):
  """Represents a specific MLIR Operation."""

  name: str
  results: List[ValueNode] = field(default_factory=list)
  operands: List[ValueNode] = field(default_factory=list)
  attributes: List[AttributeNode] = field(default_factory=list)
  regions: List[RegionNode] = field(default_factory=list)
  result_types: List[TypeNode] = field(default_factory=list)
  leading_trivia: List[TriviaNode] = field(default_factory=list)
  name_trivia: List[TriviaNode] = field(default_factory=list)
  trailing_trivia: List[TriviaNode] = field(default_factory=list)

  def to_text(self) -> str:
    parts = []

    # 1. Leading Trivia
    for t in self.leading_trivia:
      parts.append(t.to_text())

    # 2. Results
    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    # 3. Op Name
    parts.append(self.name)

    # 3b. Name Trivia
    if self.name_trivia:
      for t in self.name_trivia:
        parts.append(t.to_text())

    # 4. Operands
    if self.operands:
      # Canonical formatting: Force space before operands if no trivia exists
      if not self.name_trivia:
        parts.append(" ")
      op_names = [o.to_text() for o in self.operands]
      parts.append(f"({', '.join(op_names)})")

    # 5. Attributes
    if self.attributes:
      # Canonical formatting: Force space before attributes
      if (not self.operands) and (not self.name_trivia):
        parts.append(" ")
      elif self.operands and not self.name_trivia:
        parts.append(" ")

      parts.append("{")
      attrs_str = ", ".join([a.to_text() for a in self.attributes])
      parts.append(attrs_str)
      parts.append("}")

    # 6. Regions
    if self.regions:
      should_space = not self.name_trivia and (
        self.operands or self.attributes or (not self.operands and not self.attributes)
      )
      if should_space:
        parts.append(" ")

      for reg in self.regions:
        parts.append(reg.to_text())

    # 7. Types
    if self.result_types:
      if not self.name_trivia:
        parts.append(" : ")
      else:
        # Trivial check for trailing space in existing parts
        if parts[-1].strip() != "":
          parts.append(" : ")
        else:
          parts.append(": ")

      if len(self.result_types) == 1:
        parts.append(self.result_types[0].to_text())
      else:
        t_names = [t.to_text() for t in self.result_types]
        parts.append(f"({', '.join(t_names)})")

    # 8. Trailing Trivia
    if self.trailing_trivia:
      for t in self.trailing_trivia:
        parts.append(t.to_text())

    # Robust Newline handling
    # Ensure there is a newline at the end if one wasn't in trailing trivia rules
    if not parts or not parts[-1].endswith("\n"):
      parts.append("\n")

    return "".join(parts)


@dataclass
class ModuleNode(MlirNode):
  """Top-level container."""

  body: BlockNode

  def to_text(self) -> str:
    return self.body.to_text()
