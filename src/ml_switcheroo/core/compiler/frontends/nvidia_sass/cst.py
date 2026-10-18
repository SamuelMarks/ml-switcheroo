"""NVIDIA_SASS Concrete Syntax Tree (CST) Nodes.

Defines the strict, trivia-preserving data structures for the NVIDIA SASS syntax tree.
Each node inherits from the core `CSTNode` to guarantee exact round-trip formatting.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

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
class NvidiaSassNode(IsaNode):
  """Abstract base class for all NVIDIA_SASS CST nodes."""

  pass


@dataclass
class NvidiaSassOperand(IsaOperand, NvidiaSassNode):
  """Base class for instruction operands (Registers, Immediates, etc.)."""

  pass


@dataclass
class NvidiaSassRegister(IsaRegister, NvidiaSassOperand):
  """Represent a general-purpose register (e.g., R0, RZ).

  Attributes:
      name (str): The register identifier (e.g., "R0", "RZ").
      negated (bool): If True, prepends a negation sign (e.g., "-R0").
      absolute (bool): If True, wraps in absolute value pipes (e.g., `|R0|`).
  """

  name: str = ""
  negated: bool = False
  absolute: bool = False

  def __init__(
    self,
    name: str = "",
    negated: bool = False,
    absolute: bool = False,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassRegister.

    Args:
        name: The register identifier.
        negated: If True, prepends a negation sign.
        absolute: If True, wraps in absolute value pipes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.negated = negated
    self.absolute = absolute

  def to_text(self) -> str:
    """Render the register to its literal text.

    Returns:
        str: The rendered register text.
    """
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
class NvidiaSassPredicate(NvidiaSassOperand):
  """Represent a predicate register (e.g., @P0, !P1).

  Attributes:
      name (str): The predicate identifier (e.g., "P0", "PT").
      negated (bool): If True, indicates logical NOT (e.g., "!P0").
      is_guard (bool): If True, it is rendered with an '@' prefix.
  """

  name: str = ""
  negated: bool = False
  is_guard: bool = False

  def __init__(
    self,
    name: str = "",
    negated: bool = False,
    is_guard: bool = False,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassPredicate.

    Args:
        name: The predicate identifier.
        negated: Logical NOT flag.
        is_guard: Prefix `@` guard flag.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.negated = negated
    self.is_guard = is_guard

  def to_text(self) -> str:
    """Render the predicate to its literal text.

    Returns:
        str: The rendered predicate text.
    """
    res = "".join(t.text for t in self.leading_trivia)
    if self.is_guard:
      res += "@"
    if self.negated:
      res += "!"
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class NvidiaSassImmediate(IsaImmediate, NvidiaSassOperand):
  """Represent a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string (e.g., "0x1").
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
    """Initialize NvidiaSassImmediate.

    Args:
        value: Numeric immediate value.
        is_hex: Whether to render as hexadecimal.
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
class NvidiaSassMemory(IsaMemory, NvidiaSassOperand):
  """Represent a memory address operand.

  Supports Constant Bank access (e.g., `c[0x0][0x4]`) and Global/Local
  addressing (e.g., `[R1]`, `[R1 + 0x4]`).

  Attributes:
      base (Union[str, NvidiaSassRegister]): The base register or constant bank string.
      offset (Optional[int]): Optional byte offset to add to the base.
  """

  base: Union[str, NvidiaSassRegister] = ""
  offset: Optional[int] = None

  def __init__(
    self,
    base: Union[str, NvidiaSassRegister] = "",
    offset: Optional[int] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassMemory.

    Args:
        base: Base register or constant bank string.
        offset: Optional byte offset.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.base = base
    self.offset = offset

  def to_text(self) -> str:
    """Render the memory operand to its literal text.

    Returns:
        str: The rendered memory operand text.
    """
    res = "".join(t.text for t in self.leading_trivia)

    base_str = self.base.to_text() if isinstance(self.base, NvidiaSassRegister) else str(self.base)

    # Constant NvidiaSassMemory syntax: c[bank][offset]
    if isinstance(self.base, str) and self.base.startswith("c["):
      if self.offset is not None:
        res += f"{base_str}[{hex(self.offset)}]"
      else:
        res += f"{base_str}[0x0]"
    else:
      # NvidiaSassRegister NvidiaSassMemory syntax: [base] or [base + offset]
      if self.offset:
        res += f"[{base_str} + {hex(self.offset)}]"
      else:
        res += f"[{base_str}]"

    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class NvidiaSassInstruction(IsaInstruction, NvidiaSassNode):
  """Represent a single NVIDIA_SASS operation line.

  Attributes:
      opcode (str): The instruction mnemonic (e.g., "FADD", "MOV").
      operands (List[NvidiaSassOperand]): List of operand nodes.
      predicate (Optional[NvidiaSassPredicate]): Optional predicate guard.
  """

  opcode: str = ""
  operands: List[Any] = field(default_factory=list)
  predicate: Any = None

  def __init__(
    self,
    opcode: str = "",
    operands: Optional[List[Any]] = None,
    predicate: Any = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassInstruction.

    Args:
        opcode: Instruction mnemonic.
        operands: List of operand nodes.
        predicate: Optional predicate guard.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    self.opcode = opcode
    self.operands = operands if operands is not None else []
    self.predicate = predicate
    super().__init__(
      opcode=opcode,
      operands=operands if operands is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def __post_init__(self) -> None:
    """Validate instruction post-initialization."""
    super().__post_init__()
    if " " in self.opcode:
      raise ValueError("Invalid NVIDIA_SASS opcode")

  def to_text(self) -> str:
    """Render the instruction to its literal text.

    Returns:
        str: The rendered instruction text.
    """
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
class NvidiaSassLabel(IsaLabel, NvidiaSassOperand):
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
    """Initialize NvidiaSassLabel.

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
class NvidiaSassDirective(NvidiaSassNode):
  """Represent an assembler directive.

  Attributes:
      name (str): The directive name (e.g., "headerflags").
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
    """Initialize NvidiaSassDirective.

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
    """Render the directive to its literal text.

    Returns:
        str: The rendered directive text.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += f".{self.name}"
    if self.params:
      res += " " + ", ".join(self.params)
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class NvidiaSassComment(IsaComment, NvidiaSassNode):
  """Represent a line comment.

  Attributes:
      text (str): The comment content.
  """

  text: str = ""
  prefix: str = "//"

  def __init__(
    self,
    text: str = "",
    prefix: str = "//",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassComment.

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
class NvidiaSassModule(NvidiaSassNode):
  """Represent a complete NVIDIA_SASS code module.

  Attributes:
      statements (List[Any]): List of statements in the module.
  """

  statements: List[Any] = field(default_factory=list)

  def __init__(
    self,
    statements: Optional[List[Any]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize NvidiaSassModule.

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
    """Render the module to its literal text.

    Returns:
        str: The rendered module text.
    """
    res = "".join(t.text for t in self.leading_trivia)
    for stmt in self.statements:
      res += stmt.to_text()
    res += "".join(t.text for t in self.trailing_trivia)
    return res
