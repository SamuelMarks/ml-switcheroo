"""Common Concrete Syntax Tree (CST) Nodes for Hardware ISAs."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from ml_switcheroo.core.cst.base import CSTNode


@dataclass
class IsaNode(CSTNode):
  """Abstract base class for all ISA CST nodes."""

  pass


@dataclass
class IsaOperand(IsaNode):
  """Base class for instruction operands."""

  pass


@dataclass
class IsaRegister(IsaOperand):
  """Represent a generic register. Architectures should subclass for specifics."""

  pass


@dataclass
class IsaImmediate(IsaOperand):
  """Represent an immediate value (literal number)."""

  value: Union[int, float] = 0
  is_hex: bool = False

  def to_text(self) -> str:
    """Render the immediate value as literal text.

    Returns:
        str: Formatted hexadecimal or decimal literal text with trivia.
    """
    res = "".join(t.text for t in self.leading_trivia)
    if self.is_hex:
      res += hex(int(self.value))
    else:
      res += str(self.value)
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class IsaMemory(IsaOperand):
  """Represent a memory reference (e.g., [R0 + 4], [v0])."""

  base_reg: Optional[IsaRegister] = None
  offset: Optional[Union[int, IsaRegister]] = None

  def to_text(self) -> str:
    """Render the memory reference operand to assembly syntax.

    Returns:
        str: Bracketed memory operand string with base register and optional offset.
    """
    res = "".join(t.text for t in self.leading_trivia)
    base_str = self.base_reg.to_text().strip() if self.base_reg else ""
    res += f"[{base_str}"
    if self.offset is not None:
      if isinstance(self.offset, int):
        res += f" + {self.offset}"
      else:
        res += f" + {self.offset.to_text().strip()}"
    res += "]"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class IsaLabel(IsaNode):
  """Represent a label declaration (e.g., `LOOP_START:`)."""

  name: str = ""

  def to_text(self) -> str:
    """Render the assembly label declaration to text.

    Returns:
        str: Label identifier suffixed with a colon and surrounding trivia.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += f"{self.name}:"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class IsaComment(IsaNode):
  """Represent a comment line."""

  text: str = ""
  prefix: str = "//"

  def to_text(self) -> str:
    """Render the assembly comment line to text.

    Returns:
        str: Comment string formatted with prefix and surrounding trivia.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += f"{self.prefix} {self.text}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class IsaInstruction(IsaNode):
  """Represent a generic assembly instruction."""

  opcode: str = ""
  operands: list[Any] = field(default_factory=list)

  def to_text(self) -> str:
    """Render the assembly instruction and its comma-separated operands to text.

    Returns:
        str: Formatted assembly instruction string with surrounding trivia.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.opcode
    if self.operands:
      res += " "
      res += ", ".join(op.to_text().strip() for op in self.operands)
    res += "".join(t.text for t in self.trailing_trivia)
    return res
