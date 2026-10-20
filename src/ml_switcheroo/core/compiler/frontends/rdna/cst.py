"""RDNA Concrete Syntax Tree (CST) Nodes.

Defines the strict, trivia-preserving data structures for the AMD RDNA / GCN syntax tree.
Each node inherits from the core `CSTNode` to guarantee exact round-trip formatting.
"""

import typing

import pydantic

from ml_switcheroo.core.compiler.isa.cst import (
  IsaComment,
  IsaImmediate,
  IsaInstruction,
  IsaLabel,
  IsaMemory,
  IsaNode,
  IsaOperand,
  IsaRegister,
  TriviaInput,
)
from ml_switcheroo.core.cst.base import CSTNode


class RdnaNode(IsaNode):
  """Abstract base class for all RDNA CST nodes."""


class RdnaOperand(IsaOperand, RdnaNode):
  """Base class for instruction operands."""


class RdnaLabelRef(RdnaOperand):
  """Represent a reference to a label (e.g. as a jump target).

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaLabelRef.

    Args:
        name: Label identifier.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the label reference.

    Returns:
        str: The rendered text representation of the label reference.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaSGPR(IsaRegister, RdnaOperand):
  """Represent a Scalar General Purpose Register.

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., s[0:3]).
  """

  index: int = 0
  count: int = 1

  def __init__(
    self,
    index: int = 0,
    count: int = 1,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaSGPR.

    Args:
        index: Starting register index.
        count: Register count.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      index=index,
      count=count,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the RdnaSGPR.

    Returns:
        str: The rendered text representation of the SGPR.
    """
    res = "".join(t.text for t in self.leading_trivia)
    if self.count > 1:
      end = self.index + self.count - 1
      res += f"s[{self.index}:{end}]"
    else:
      res += f"s{self.index}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaVGPR(IsaRegister, RdnaOperand):
  """Represent a Vector General Purpose Register.

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., v[0:3]).
  """

  index: int = 0
  count: int = 1

  def __init__(
    self,
    index: int = 0,
    count: int = 1,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaVGPR.

    Args:
        index: Starting register index.
        count: Register count.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      index=index,
      count=count,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the RdnaVGPR.

    Returns:
        str: The rendered text representation of the VGPR.
    """
    res = "".join(t.text for t in self.leading_trivia)
    if self.count > 1:
      end = self.index + self.count - 1
      res += f"v[{self.index}:{end}]"
    else:
      res += f"v{self.index}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


def c_SGPR(idx: int) -> RdnaSGPR:
  """Support to create a single RdnaSGPR.

  Args:
      idx (int): The register index.

  Returns:
      RdnaSGPR: A new RdnaSGPR instance.
  """
  return RdnaSGPR(index=idx)


def c_VGPR(idx: int) -> RdnaVGPR:
  """Support to create a single RdnaVGPR.

  Args:
      idx (int): The register index.

  Returns:
      RdnaVGPR: A new RdnaVGPR instance.
  """
  return RdnaVGPR(index=idx)


class RdnaImmediate(IsaImmediate, RdnaOperand):
  """Represent a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string.
  """

  value: typing.Union[int, float] = 0
  is_hex: bool = False

  def __init__(
    self,
    value: typing.Union[int, float] = 0,
    is_hex: bool = False,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaImmediate.

    Args:
        value: Numeric value.
        is_hex: Whether to render as hex.
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


class RdnaModifier(RdnaOperand):
  """Represent an instruction modifier or attribute (e.g., glc, off).

  Attributes:
      name (str): The modifier string.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaModifier.

    Args:
        name: Modifier string.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the modifier.

    Returns:
        str: The rendered text representation of the modifier.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaMemory(IsaMemory, RdnaOperand):
  """Represent a memory address operand.

  Attributes:
      base (Union[RdnaSGPR, RdnaVGPR]): The base register.
      offset (Optional[int]): RdnaImmediate byte offset.
  """

  base: typing.Union[RdnaSGPR, RdnaVGPR, IsaRegister, str] = pydantic.Field(default_factory=lambda: c_VGPR(0))
  offset: typing.Optional[int] = None

  def __init__(
    self,
    base: typing.Optional[typing.Union[RdnaSGPR, RdnaVGPR, IsaRegister, str]] = None,
    offset: typing.Optional[int] = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaMemory.

    Args:
        base: Base register.
        offset: Byte offset.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.base = base if base is not None else c_VGPR(0)
    self.offset = offset

  def to_text(self) -> str:
    """Render the memory operand.

    Returns:
        str: The rendered text representation of the memory address.
    """
    res = "".join(t.text for t in self.leading_trivia)
    base_str = self.base.to_text() if isinstance(self.base, CSTNode) else str(self.base)
    res += base_str
    if self.offset is not None and self.offset != 0:
      res += f" offset:{self.offset}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaInstruction(IsaInstruction, RdnaNode):
  """Represent a single RDNA operation line.

  Attributes:
      opcode (str): The instruction mnemonic.
      operands (List[RdnaOperand]): List of operand nodes.
  """

  opcode: str = ""
  operands: typing.Annotated[typing.List[RdnaNode], pydantic.SkipValidation] = pydantic.Field(default_factory=list)  # type: ignore[assignment]

  def __init__(
    self,
    opcode: str = "",
    operands: typing.Optional[typing.Sequence[typing.Union[RdnaNode, IsaNode]]] = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaInstruction.

    Args:
        opcode: Instruction mnemonic.
        operands: List of operand nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      opcode=opcode,
      operands=list(operands) if operands is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def __post_init__(self) -> None:
    """Validate instruction post-initialization.

    Raises:
        ValueError: If the opcode contains spaces.
    """
    super().__post_init__()
    if " " in self.opcode:
      raise ValueError("Invalid RDNA opcode")

  def to_text(self) -> str:
    """Render the instruction.

    Returns:
        str: The rendered text representation of the instruction.
    """
    res = "".join(t.text for t in self.leading_trivia)
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

    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaLabel(IsaLabel, RdnaNode):
  """Represent a jump target label.

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaLabel.

    Args:
        name: Label identifier.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )


class RdnaDirective(RdnaNode):
  """Represent an assembler directive.

  Attributes:
      name (str): The directive name.
      params (List[str]): List of string parameters.
  """

  name: str = ""
  params: typing.List[str] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    params: typing.Optional[typing.List[str]] = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaDirective.

    Args:
        name: Directive name.
        params: Optional list of string parameters.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      params=params if params is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the directive.

    Returns:
        str: The rendered text representation of the directive.
    """
    res = "".join(t.text for t in self.leading_trivia)
    out = f".{self.name}"
    if self.params:
      out += " " + ", ".join(self.params)
    res += out
    res += "".join(t.text for t in self.trailing_trivia)
    return res


class RdnaComment(IsaComment, RdnaNode):
  """Represent a line comment.

  Attributes:
      text (str): The comment content.
  """

  text: str = ""
  prefix: str = ";"

  def __init__(
    self,
    text: str = "",
    prefix: str = ";",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaComment.

    Args:
        text: Comment text.
        prefix: Comment prefix.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      text=text,
      prefix=prefix,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )


class RdnaModule(RdnaNode):
  """Represent a complete RDNA code module.

  Attributes:
      statements (List[RdnaNode]): List of statements.
  """

  statements: typing.Annotated[typing.List[RdnaNode], pydantic.SkipValidation] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    statements: typing.Optional[typing.Sequence[typing.Union[RdnaNode, IsaNode]]] = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RdnaModule.

    Args:
        statements: List of statements in the module.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      statements=list(statements) if statements is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Render the module.

    Returns:
        str: The rendered text representation of the entire module.
    """
    res = "".join(t.text for t in self.leading_trivia)
    for stmt in self.statements:
      res += stmt.to_text()
    res += "".join(t.text for t in self.trailing_trivia)
    return res
