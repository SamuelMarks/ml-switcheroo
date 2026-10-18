"""RDNA Concrete Syntax Tree (CST) Nodes.

Defines the strict, trivia-preserving data structures for the AMD RDNA / GCN syntax tree.
Each node inherits from the core `CSTNode` to guarantee exact round-trip formatting.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ml_switcheroo.core.compiler.isa.cst import (
  IsaComment,
  IsaImmediate,
  IsaInstruction,
  IsaLabel,
  IsaMemory,
  IsaNode,
  IsaOperand,
  IsaRegister,
)


@dataclass
class RdnaNode(IsaNode):
  """Abstract base class for all RDNA CST nodes."""

  pass


@dataclass
class RdnaOperand(IsaOperand, RdnaNode):
  """Base class for instruction operands."""

  pass


@dataclass
class RdnaLabelRef(RdnaOperand):
  """Represent a reference to a label (e.g. as a jump target).

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaLabelRef.

    Args:
        name: Label identifier.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name

  def to_text(self) -> str:
    """Render the label reference.

    Returns:
        str: The rendered text representation of the label reference.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
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
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaSGPR.

    Args:
        index: Starting register index.
        count: Register count.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.index = index
    self.count = count

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


@dataclass
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
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaVGPR.

    Args:
        index: Starting register index.
        count: Register count.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.index = index
    self.count = count

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


@dataclass
class RdnaImmediate(IsaImmediate, RdnaOperand):
  """Represent a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string.
  """

  value: Any = 0
  is_hex: bool = False

  def __init__(
    self,
    value: Any = 0,
    is_hex: bool = False,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaImmediate.

    Args:
        value: Numeric value.
        is_hex: Whether to render as hex.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      value=value,
      is_hex=is_hex,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )


@dataclass
class RdnaModifier(RdnaOperand):
  """Represent an instruction modifier or attribute (e.g., glc, off).

  Attributes:
      name (str): The modifier string.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaModifier.

    Args:
        name: Modifier string.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name

  def to_text(self) -> str:
    """Render the modifier.

    Returns:
        str: The rendered text representation of the modifier.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaMemory(IsaMemory, RdnaOperand):
  """Represent a memory address operand.

  Attributes:
      base (Union[RdnaSGPR, RdnaVGPR]): The base register.
      offset (Optional[int]): RdnaImmediate byte offset.
  """

  base: Any = field(default_factory=lambda: c_VGPR(0))
  offset: Optional[int] = None

  def __init__(
    self,
    base: Any = None,
    offset: Optional[int] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaMemory.

    Args:
        base: Base register.
        offset: Byte offset.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
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
    base_str = self.base.to_text() if hasattr(self.base, "to_text") else str(self.base)
    res += base_str
    if self.offset is not None and self.offset != 0:
      res += f" offset:{self.offset}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaInstruction(IsaInstruction, RdnaNode):
  """Represent a single RDNA operation line.

  Attributes:
      opcode (str): The instruction mnemonic.
      operands (List[RdnaOperand]): List of operand nodes.
  """

  opcode: str = ""
  operands: List[Any] = field(default_factory=list)

  def __init__(
    self,
    opcode: str = "",
    operands: Optional[List[Any]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaInstruction.

    Args:
        opcode: Instruction mnemonic.
        operands: List of operand nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    self.opcode = opcode
    self.operands = operands if operands is not None else []
    super().__init__(
      opcode=opcode,
      operands=operands if operands is not None else [],
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


@dataclass
class RdnaLabel(IsaLabel, RdnaNode):
  """Represent a jump target label.

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def __init__(
    self,
    name: str = "",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaLabel.

    Args:
        name: Label identifier.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      name=name,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name


@dataclass
class RdnaDirective(RdnaNode):
  """Represent an assembler directive.

  Attributes:
      name (str): The directive name.
      params (List[str]): List of string parameters.
  """

  name: str = ""
  params: List[str] = field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    params: Optional[List[str]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaDirective.

    Args:
        name: Directive name.
        params: Optional list of string parameters.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.params = params if params is not None else []

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


@dataclass
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
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaComment.

    Args:
        text: Comment text.
        prefix: Comment prefix.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      text=text,
      prefix=prefix,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.text = text
    self.prefix = prefix


@dataclass
class RdnaModule(RdnaNode):
  """Represent a complete RDNA code module.

  Attributes:
      statements (List[Any]): List of statements.
  """

  statements: List[Any] = field(default_factory=list)

  def __init__(
    self,
    statements: Optional[List[Any]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RdnaModule.

    Args:
        statements: List of statements in the module.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.statements = statements if statements is not None else []

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
