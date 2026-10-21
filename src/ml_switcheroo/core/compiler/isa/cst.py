"""Common Concrete Syntax Tree (CST) Nodes for Hardware ISAs."""

import typing

import pydantic

from ml_switcheroo.core.cst.base import CSTNode, Trivia

TriviaInput: typing.TypeAlias = list[Trivia] | str | None


class IsaNode(CSTNode):
  """Abstract base class for all ISA CST nodes."""


class IsaOperand(IsaNode):
  """Base class for instruction operands."""


class IsaRegister(IsaOperand):
  """Represent a generic register. Architectures should subclass for specifics."""


class IsaImmediate(IsaOperand):
  """Represent an immediate value (literal number)."""

  value: int | float = 0
  is_hex: bool = False

  def __init__(
    self,
    value: int | float = 0,
    is_hex: bool = False,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize IsaImmediate.

    Args:
        value: Numeric immediate value.
        is_hex: Whether to format as hexadecimal.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      value=value,
      is_hex=is_hex,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class IsaMemory(IsaOperand):
  """Represent a memory reference (e.g., [R0 + 4], [v0])."""

  base_reg: IsaRegister | None = None
  offset: int | IsaRegister | None = None

  def __init__(
    self,
    base_reg: IsaRegister | None = None,
    offset: int | IsaRegister | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize IsaMemory.

    Args:
        base_reg: Optional base register.
        offset: Optional integer or register offset.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      base_reg=base_reg,
      offset=offset,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class IsaInstruction(IsaNode):
  """Represent a generic assembly instruction."""

  opcode: str = ""
  operands: list[IsaNode] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    opcode: str = "",
    operands: list[IsaNode] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize IsaInstruction.

    Args:
        opcode: Instruction mnemonic.
        operands: List of operand nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      opcode=opcode,
      operands=operands if operands is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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
