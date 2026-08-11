"""SASS Concrete Syntax Tree (CST) Nodes.

Defines the strict, trivia-preserving data structures for the NVIDIA SASS syntax tree.
Each node inherits from the core `CSTNode` to guarantee exact round-trip formatting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ml_switcheroo.core.cst.base import CSTNode


@dataclass
class SassNode(CSTNode):
  """Abstract base class for all SASS CST nodes."""

  pass


@dataclass
class SassOperand(SassNode):
  """Base class for instruction operands (Registers, Immediates, etc.)."""

  pass


@dataclass
class SassRegister(SassOperand):
  """Represents a general-purpose register (e.g., R0, RZ).

  Attributes:
      name (str): The register identifier (e.g., "R0", "RZ").
      negated (bool): If True, prepends a negation sign (e.g., "-R0").
      absolute (bool): If True, wraps in absolute value pipes (e.g., `|R0|`).
  """

  name: str = ""
  negated: bool = False
  absolute: bool = False

  def to_text(self) -> str:
    """Renders the register to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    if self.negated:
      res += "-"
    if self.absolute:
      res += f"|{self.name}|"
    else:
      res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassPredicate(SassOperand):
  """Represents a predicate register (e.g., @P0, !P1).

  Attributes:
      name (str): The predicate identifier (e.g., "P0", "PT").
      negated (bool): If True, indicates logical NOT (e.g., "!P0").
      is_guard (bool): If True, it is rendered with an '@' prefix.
  """

  name: str = ""
  negated: bool = False
  is_guard: bool = False

  def to_text(self) -> str:
    """Renders the predicate to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    if self.is_guard:
      res += "@"
    if self.negated:
      res += "!"
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassImmediate(SassOperand):
  """Represents a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string (e.g., "0x1").
  """

  value: Union[int, float] = 0
  is_hex: bool = False

  def to_text(self) -> str:
    """Renders the immediate to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    if self.is_hex:
      if isinstance(self.value, float):
        res += hex(int(self.value))
      else:
        res += hex(int(self.value))
    else:
      res += str(self.value)
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassMemory(SassOperand):
  """Represents a memory address operand.

  Supports Constant Bank access (e.g., `c[0x0][0x4]`) and Global/Local
  addressing (e.g., `[R1]`, `[R1 + 0x4]`).

  Attributes:
      base (Union[str, SassRegister]): The base register or constant bank string.
      offset (Optional[int]): Optional byte offset to add to the base.
  """

  base: Union[str, SassRegister] = ""
  offset: Optional[int] = None

  def to_text(self) -> str:
    """Renders the memory operand to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)

    base_str = self.base.to_text() if isinstance(self.base, SassRegister) else str(self.base)

    # Constant SassMemory syntax: c[bank][offset]
    if isinstance(self.base, str) and self.base.startswith("c["):
      if self.offset is not None:
        res += f"{base_str}[{hex(self.offset)}]"
      else:
        res += f"{base_str}[0x0]"
    else:
      # SassRegister SassMemory syntax: [base] or [base + offset]
      if self.offset:
        res += f"[{base_str} + {hex(self.offset)}]"
      else:
        res += f"[{base_str}]"

    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassInstruction(SassNode):
  """Represents a single SASS operation line.

  Attributes:
      opcode (str): The instruction mnemonic (e.g., "FADD", "MOV").
      operands (List[SassOperand]): List of operand nodes.
      predicate (Optional[SassPredicate]): Optional predicate guard.
  """

  opcode: str = ""
  operands: List[SassOperand] = field(default_factory=list)
  predicate: Optional[SassPredicate] = None

  def __post_init__(self) -> None:
    """Validate instruction post-initialization."""
    if " " in self.opcode:
      raise ValueError("Invalid SASS opcode")

  def to_text(self) -> str:
    """Renders the instruction to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    if self.predicate:
      pred_text = self.predicate.to_text()
      res += pred_text
      if not self.leading_trivia and not self.predicate.trailing_trivia:
        res += " "

    res += self.opcode

    for i, op in enumerate(self.operands):
      op_text = op.to_text()
      if not op.leading_trivia:
        if i > 0:
          res += ", " + op_text
        else:
          res += " " + op_text
      else:
        res += op_text

    if not self.trailing_trivia:
      res += ";"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassLabel(SassOperand):
  """Represents a jump target label.

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def to_text(self) -> str:
    """Renders the label to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    res += f"{self.name}:"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassDirective(SassNode):
  """Represents an assembler directive.

  Attributes:
      name (str): The directive name (e.g., "headerflags").
      params (List[str]): List of string parameters.
  """

  name: str = ""
  params: List[str] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the directive to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    res += f".{self.name}"
    if self.params:
      res += " " + ", ".join(self.params)
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassComment(SassNode):
  """Represents a line comment.

  Attributes:
      text (str): The comment content.
  """

  text: str = ""

  def to_text(self) -> str:
    """Renders the comment to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    res += f"// {self.text}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class SassModule(SassNode):
  """Represents a complete SASS code module.

  Attributes:
      statements (List[SassNode]): List of statements in the module.
  """

  statements: List[SassNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the module to its literal text."""
    res = "".join(t.text for t in self.leading_trivia)
    for stmt in self.statements:
      res += stmt.to_text()
    res += "".join(t.text for t in self.trailing_trivia)
    return res
