"""Common Concrete Syntax Tree (CST) Nodes for Hardware ISAs."""

import typing

import pydantic

from ml_switcheroo.core.cst.base import (
  CstFieldValue as CstFieldValue,
)
from ml_switcheroo.core.cst.base import (
  CSTNode,
)
from ml_switcheroo.core.cst.base import (
  Trivia as Trivia,
)
from ml_switcheroo.core.cst.base import (
  TriviaInput as TriviaInput,
)


class IsaNode(CSTNode):
  """Abstract base class for all ISA CST nodes."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaNode.

    Args:
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to CSTNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)


class IsaOperand(IsaNode):
  """Base class for instruction operands."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaOperand.

    Args:
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to IsaNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)


class IsaRegister(IsaOperand):
  """Represent a generic register. Architectures should subclass for specifics."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaRegister.

    Args:
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to IsaOperand.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)


class IsaImmediate(IsaOperand):
  """Represent an immediate value (literal number).

  Attributes:
      value: The value (`int | float`).
      is_hex: The is hex (`bool`).
  """

  value: int | float = 0
  is_hex: bool = False

  def __init__(
    self,
    value: int | float = 0,
    is_hex: bool = False,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaImmediate.

    Args:
        value: Numeric immediate value.
        is_hex: Whether to format as hexadecimal.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      value=value,
      is_hex=is_hex,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
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
  """Represent a memory reference (e.g., [R0 + 4], [v0]).

  Attributes:
      base_reg: The base reg (`IsaRegister | None`).
      offset: The offset (`int | IsaRegister | None`).
  """

  base_reg: IsaRegister | None = None
  offset: int | IsaRegister | None = None

  def __init__(
    self,
    base_reg: IsaRegister | None = None,
    offset: int | IsaRegister | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaMemory.

    Args:
        base_reg: Optional base register.
        offset: Optional integer or register offset.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      base_reg=base_reg,
      offset=offset,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
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
  """Represent a label declaration (e.g., `LOOP_START:`).

  Attributes:
      name: The name (`str`).
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaLabel.

    Args:
        name: Label identifier.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      name=name,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
    )

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
  """Represent a comment line.

  Attributes:
      text: The text (`str`).
      prefix: The prefix (`str`).
  """

  text: str = ""
  prefix: str = "//"

  def __init__(
    self,
    text: str = "",
    prefix: str = "//",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaComment.

    Args:
        text: Comment text.
        prefix: Comment prefix.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      text=text,
      prefix=prefix,
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
    )

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
  """Represent a generic assembly instruction.

  Attributes:
      opcode: The opcode (`str`).
      operands: The operands (`list[IsaNode]`).
  """

  opcode: str = ""
  operands: list[IsaNode] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    opcode: str = "",
    operands: list[IsaNode] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaInstruction.

    Args:
        opcode: Instruction mnemonic.
        operands: List of operand nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      opcode=opcode,
      operands=operands if operands is not None else [],
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
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


class IsaDirective(IsaNode):
  """Represent an assembler directive (e.g., ``.headerflags``, ``.text``).

  Attributes:
      name: Directive identifier without the leading dot (`str`).
      params: Directive parameter strings (`list[str]`).
  """

  name: str = ""
  params: list[str] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    params: list[str] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaDirective.

    Args:
        name: Directive name.
        params: Optional list of string parameters.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to IsaNode.
    """
    super().__init__(
      name=name,
      params=params if params is not None else [],
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
    )

  def to_text(self) -> str:
    """Render the directive to its literal assembly text.

    Returns:
        str: Rendered directive text with surrounding trivia.
    """
    res = self.leading_trivia_text()
    res += f".{self.name}"
    if self.params:
      res += " " + ", ".join(self.params)
    res += self.trailing_trivia_text()
    return res


IsaStmtT = typing.TypeVar("IsaStmtT", bound=IsaNode)


class IsaModule(IsaNode, typing.Generic[IsaStmtT]):
  """Represent a complete hardware ISA assembly module.

  Attributes:
      statements: Ordered list of statement CST nodes (`list[IsaStmtT]`).
  """

  statements: list[IsaStmtT] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    statements: typing.Sequence[IsaStmtT] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize IsaModule.

    Args:
        statements: Ordered statement nodes comprising the module.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to IsaNode.
    """
    super().__init__(
      statements=list(statements) if statements is not None else [],
      leading_trivia=leading_trivia,
      trailing_trivia=trailing_trivia,
      **kwargs,
    )

  def to_text(self) -> str:
    """Render the module statements and surrounding trivia to literal assembly text.

    Returns:
        str: Full rendered assembly module text.
    """
    res = self.leading_trivia_text()
    for stmt in self.statements:
      res += stmt.to_text()
    res += self.trailing_trivia_text()
    return res
