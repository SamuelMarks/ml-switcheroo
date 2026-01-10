"""
RDNA AST Nodes.

Defines the data structures for the AMD RDNA / GCN syntax tree.
Each node corresponds to a syntactic element in RDNA assembly and
implements `__str__` to emit valid assembly code.
"""

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Union


class RdnaNode(abc.ABC):
  """Abstract base class for all RDNA AST nodes."""

  @abc.abstractmethod
  def __str__(self) -> str:
    """Returns the valid RDNA string representation of the node."""
    pass


@dataclass
class Operand(RdnaNode):
  """Base class for instruction operands (Registers, Immediates, etc.)."""

  def __str__(self) -> str:
    return ""


@dataclass
class LabelRef(Operand):
  """
  Represents a reference to a label (e.g. as a jump target) or generic identifier operand.

  Attributes:
      name (str): The label identifier.
  """

  name: str

  def __str__(self) -> str:
    return self.name


@dataclass
class SGPR(Operand):
  """
  Represents a Scalar General Purpose Register (e.g., s0, s10).

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., s[0:3]).
  """

  index: int
  count: int = 1

  def __str__(self) -> str:
    if self.count > 1:
      end = self.index + self.count - 1
      return f"s[{self.index}:{end}]"
    return f"s{self.index}"


@dataclass
class VGPR(Operand):
  """
  Represents a Vector General Purpose Register (e.g., v0, v255).

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., v[0:3]).
  """

  index: int
  count: int = 1

  def __str__(self) -> str:
    if self.count > 1:
      end = self.index + self.count - 1
      return f"v[{self.index}:{end}]"
    return f"v{self.index}"


# Helper aliases for creating registers
def c_SGPR(idx: int) -> SGPR:
  """Helper to create a single SGPR."""
  return SGPR(idx)


def c_VGPR(idx: int) -> VGPR:
  """Helper to create a single VGPR."""
  return VGPR(idx)


@dataclass
class Immediate(Operand):
  """
  Represents a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string (e.g., 0xff).
  """

  value: Union[int, float]
  is_hex: bool = False

  def __str__(self) -> str:
    if self.is_hex:
      return hex(int(self.value))
    return str(self.value)


@dataclass
class Modifier(Operand):
  """
  Represents an instruction modifier or attribute.
  e.g., `glc`, `slc`, `off`.

  Attributes:
      name (str): The modifier string.
  """

  name: str

  def __str__(self) -> str:
    return self.name


@dataclass
class Memory(Operand):
  """
  Represents a memory address operand, typically used in generic loads/stores.
  RDNA often passes the address as a register or register pair, but explicit
  offset syntax exists.

  Format: `method[base + offset]` or simply register references depending on op.
  This node models the bracketed syntax `global_load_dword v0, v[1:2], off`.
  Usually, RDNA just uses registers as operands, but sometimes specific addressing notation is used.
  Here we treat it as an explicit container if needed, primarily for offsets.

  Attributes:
      base (Union[SGPR, VGPR]): The base register.
      offset (Optional[int]): Immediate byte offset.
  """

  base: Union[SGPR, VGPR]
  offset: Optional[int] = None

  def __str__(self) -> str:
    base_str = str(self.base)
    if self.offset is not None and self.offset != 0:
      return f"{base_str} offset:{self.offset}"
    return base_str


@dataclass
class Instruction(RdnaNode):
  """
  Represents a single RDNA operation line.

  Format: `{opcode} {operands}`
  Modifiers are treated as operands in the list for flexible placement.

  Attributes:
      opcode (str): The instruction mnemonic (e.g., "v_add_f32", "s_mov_b32").
      operands (List[Operand]): List of operand nodes.
  """

  opcode: str
  operands: List[Operand] = field(default_factory=list)

  def __str__(self) -> str:
    if not self.operands:
      return self.opcode
    # Modifiers often appear without commas in some diassemblies, but standard
    # assembly usually comma-separates or space-separates depending on the specific assembler version.
    # We default to comma-separated for standard operands.
    # Special handling: modifiers often just append at the end.
    parts = []
    for op in self.operands:
      parts.append(str(op))
    ops_str = ", ".join(parts)
    return f"{self.opcode} {ops_str}"


@dataclass
class Label(RdnaNode):
  """
  Represents a jump target label.

  Format: `{name}:`

  Attributes:
      name (str): The label identifier.
  """

  name: str

  def __str__(self) -> str:
    return f"{self.name}:"


@dataclass
class Directive(RdnaNode):
  """
  Represents an assembler directive.

  Format: `.{name} {params}`

  Attributes:
      name (str): The directive name (e.g., "global_base").
      params (List[str]): List of string parameters.
  """

  name: str
  params: List[str] = field(default_factory=list)

  def __str__(self) -> str:
    out = f".{self.name}"
    if self.params:
      out += " " + ", ".join(self.params)
    return out


@dataclass
class Comment(RdnaNode):
  """
  Represents a line comment.
  RDNA assembly uses `;` for comments.

  Format: `; {text}`

  Attributes:
      text (str): The comment content.
  """

  text: str

  def __str__(self) -> str:
    return f"; {self.text}"
