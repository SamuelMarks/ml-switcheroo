"""
SASS AST Nodes.

Defines the data structures for the NVIDIA SASS syntax tree.
Each node corresponds to a syntactic element in SASS assembly and
implements `__str__` to emit valid code.
"""

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Union


class SassNode(abc.ABC):
  """Abstract base class for all SASS AST nodes."""

  @abc.abstractmethod
  def __str__(self) -> str:
    """Returns the valid SASS string representation of the node."""
    pass


@dataclass
class Operand(SassNode):
  """Base class for instruction operands (Registers, Immediates, etc.)."""

  pass


@dataclass
class Register(Operand):
  """
  Represents a general-purpose register (e.g., R0, RZ).

  Attributes:
      name (str): The register identifier (e.g., "R0", "RZ").
      negated: If True, prepends a negation sign (e.g., "-R0").
      absolute: If True, wraps in absolute value pipes (e.g., "|R0|").
  """

  name: str
  negated: bool = False
  absolute: bool = False

  def __str__(self) -> str:
    res = self.name
    if self.absolute:
      res = f"|{res}|"
    if self.negated:
      res = f"-{res}"
    return res


@dataclass
class Predicate(Operand):
  """
  Represents a predicate register (e.g., @P0, !P1).

  Used both as instruction guards (preceding the opcode) and as operands
  in logical instructions (ISETP).

  Attributes:
      name (str): The predicate identifier (e.g., "P0", "PT").
      negated (bool): If True, indicates logical NOT (e.g., "!P0").
  """

  name: str
  negated: bool = False

  def __str__(self) -> str:
    prefix = "!" if self.negated else ""
    return f"{prefix}{self.name}"


@dataclass
class Immediate(Operand):
  """
  Represents a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string (e.g., "0x1").
  """

  value: Union[int, float]
  is_hex: bool = False

  def __str__(self) -> str:
    if self.is_hex:
      if isinstance(self.value, float):
        # Float hex representation requires struct packing usually,
        # but SASS often takes raw hex bytes for float encoding.
        # Here we assume user passes int representation of float bits if hex desired.
        return hex(int(self.value))
      return hex(int(self.value))
    return str(self.value)


@dataclass
class Memory(Operand):
  """
  Represents a memory address operand.

  Supports Constant Bank access (e.g., `c[0x0][0x4]`) and Global/Local
  addressing (e.g., `[R1]`, `[R1 + 0x4]`).

  Attributes:
      base (Union[str, Register]): The base register (e.g., "R1") or constant bank string (e.g., "c[0x0]").
      offset (Optional[int]): Optional byte offset to add to the base.
  """

  base: Union[str, Register]
  offset: Optional[int] = None

  def __str__(self) -> str:
    base_str = str(self.base)
    # Constant Memory syntax: c[bank][offset]
    if isinstance(self.base, str) and self.base.startswith("c["):
      if self.offset is not None:
        return f"{base_str}[{hex(self.offset)}]"
      return f"{base_str}[0x0]"

    # Register Memory syntax: [base] or [base + offset]
    if self.offset:
      return f"[{base_str} + {hex(self.offset)}]"
    return f"[{base_str}]"


@dataclass
class Instruction(SassNode):
  """
  Represents a single SASS operation line.

  Format: `@{predicate} {opcode} {operands};`

  Attributes:
      opcode (str): The instruction mnemonic (e.g., "FADD", "MOV").
      operands (List[Operand]): List of operand nodes.
      predicate (Optional[Predicate]): Optional predicate guard executed before instruction.
  """

  opcode: str
  operands: List[Operand] = field(default_factory=list)
  predicate: Optional[Predicate] = None

  def __str__(self) -> str:
    pred_str = f"@{str(self.predicate)} " if self.predicate else ""
    ops_str = ", ".join(str(op) for op in self.operands)
    return f"{pred_str}{self.opcode} {ops_str};"


@dataclass
class Label(SassNode):
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
class Directive(SassNode):
  """
  Represents an assembler directive.

  Format: `.{name} {params}`

  Attributes:
      name (str): The directive name (e.g., "headerflags").
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
class Comment(SassNode):
  """
  Represents a line comment.

  Format: `// {text}`

  Attributes:
      text: The comment content.
  """

  text: str

  def __str__(self) -> str:
    return f"// {self.text}"


# Compat alias: SGPR is not native in SASS but needed for macros sharing patterns if any
class SGPR(Operand):
  pass
