"""RDNA Concrete Syntax Tree (CST) Nodes.

Defines the strict, trivia-preserving data structures for the AMD RDNA / GCN syntax tree.
Each node inherits from the core `CSTNode` to guarantee exact round-trip formatting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ml_switcheroo.core.cst.base import CSTNode


@dataclass
class RdnaNode(CSTNode):
  """Abstract base class for all RDNA CST nodes."""

  pass


@dataclass
class RdnaOperand(RdnaNode):
  """Base class for instruction operands."""

  pass


@dataclass
class RdnaLabelRef(RdnaOperand):
  """Represents a reference to a label (e.g. as a jump target).

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def to_text(self) -> str:
    """Renders the label reference.

    Returns:
        str: The rendered text representation of the label reference.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaSGPR(RdnaOperand):
  """Represents a Scalar General Purpose Register.

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., s[0:3]).
  """

  index: int = 0
  count: int = 1

  def to_text(self) -> str:
    """Renders the RdnaSGPR.

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
class RdnaVGPR(RdnaOperand):
  """Represents a Vector General Purpose Register.

  Attributes:
      index (int): The register index.
      count (int): If > 1, represents a range/multi-register (e.g., v[0:3]).
  """

  index: int = 0
  count: int = 1

  def to_text(self) -> str:
    """Renders the RdnaVGPR.

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
  """Helper to create a single RdnaSGPR.

  Args:
      idx (int): The register index.

  Returns:
      RdnaSGPR: A new RdnaSGPR instance.
  """
  return RdnaSGPR(index=idx)


def c_VGPR(idx: int) -> RdnaVGPR:
  """Helper to create a single RdnaVGPR.

  Args:
      idx (int): The register index.

  Returns:
      RdnaVGPR: A new RdnaVGPR instance.
  """
  return RdnaVGPR(index=idx)


@dataclass
class RdnaImmediate(RdnaOperand):
  """Represents a literal constant value.

  Attributes:
      value (Union[int, float]): The numeric value.
      is_hex (bool): If True, renders as hex string.
  """

  value: Union[int, float] = 0
  is_hex: bool = False

  def to_text(self) -> str:
    """Renders the immediate.

    Returns:
        str: The rendered text representation of the immediate value.
    """
    res = "".join(t.text for t in self.leading_trivia)
    if self.is_hex:
      res += hex(int(self.value))
    else:
      res += str(self.value)
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaModifier(RdnaOperand):
  """Represents an instruction modifier or attribute (e.g., glc, off).

  Attributes:
      name (str): The modifier string.
  """

  name: str = ""

  def to_text(self) -> str:
    """Renders the modifier.

    Returns:
        str: The rendered text representation of the modifier.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += self.name
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaMemory(RdnaOperand):
  """Represents a memory address operand.

  Attributes:
      base (Union[RdnaSGPR, RdnaVGPR]): The base register.
      offset (Optional[int]): RdnaImmediate byte offset.
  """

  base: Union[RdnaSGPR, RdnaVGPR] = field(default_factory=lambda: c_VGPR(0))
  offset: Optional[int] = None

  def to_text(self) -> str:
    """Renders the memory operand.

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
class RdnaInstruction(RdnaNode):
  """Represents a single RDNA operation line.

  Attributes:
      opcode (str): The instruction mnemonic.
      operands (List[RdnaOperand]): List of operand nodes.
  """

  opcode: str = ""
  operands: List[RdnaOperand] = field(default_factory=list)

  def __post_init__(self) -> None:
    """Validate instruction post-initialization.

    Raises:
        ValueError: If the opcode contains spaces.
    """
    if " " in self.opcode:
      raise ValueError("Invalid RDNA opcode")

  def to_text(self) -> str:
    """Renders the instruction.

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
class RdnaLabel(RdnaNode):
  """Represents a jump target label.

  Attributes:
      name (str): The label identifier.
  """

  name: str = ""

  def to_text(self) -> str:
    """Renders the label.

    Returns:
        str: The rendered text representation of the label.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += f"{self.name}:"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaDirective(RdnaNode):
  """Represents an assembler directive.

  Attributes:
      name (str): The directive name.
      params (List[str]): List of string parameters.
  """

  name: str = ""
  params: List[str] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the directive.

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
class RdnaComment(RdnaNode):
  """Represents a line comment.

  Attributes:
      text (str): The comment content.
  """

  text: str = ""

  def to_text(self) -> str:
    """Renders the comment.

    Returns:
        str: The rendered text representation of the comment.
    """
    res = "".join(t.text for t in self.leading_trivia)
    res += f"; {self.text}"
    res += "".join(t.text for t in self.trailing_trivia)
    return res


@dataclass
class RdnaModule(RdnaNode):
  """Represents a complete RDNA code module.

  Attributes:
      statements (List[RdnaNode]): List of statements.
  """

  statements: List[RdnaNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the module.

    Returns:
        str: The rendered text representation of the entire module.
    """
    res = "".join(t.text for t in self.leading_trivia)
    for stmt in self.statements:
      res += stmt.to_text()
    res += "".join(t.text for t in self.trailing_trivia)
    return res
