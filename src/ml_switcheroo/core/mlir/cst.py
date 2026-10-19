"""MLIR Concrete Syntax Tree Nodes.

This module defines the data structures for representing MLIR source code.
It ensures structural hierarchy (Module -> Operation -> Region -> Block)
and trivia preservation for high-fidelity round-tripping.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ml_switcheroo.core.cst.base import CSTNode, Trivia


@dataclass
class MlirNode(CSTNode):
  """Abstract base class for all MLIR CST nodes."""

  pass


@dataclass
class TypeNode(MlirNode):
  """Represent a type annotation."""

  body: str = ""

  def __init__(
    self,
    body: str = "",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TypeNode.

    Args:
        body: Type string body.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.body = body

  def to_text(self) -> str:
    """Return textual representation.

    Returns:
        str: The rendered type string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += self.body
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class ValueNode(MlirNode):
  """Represent an SSA Value identifier (e.g. %0)."""

  name: str = ""
  use_index: Optional[int] = None
  type_node: Optional[TypeNode] = None
  colon_trivia: List[Trivia] = field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    use_index: Optional[int] = None,
    type_node: Optional[TypeNode] = None,
    colon_trivia: Optional[List[Trivia]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize ValueNode.

    Args:
        name: SSA value identifier.
        use_index: Optional result index.
        type_node: Optional type annotation node.
        colon_trivia: Optional trivia preceding colon.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.use_index = use_index
    self.type_node = type_node
    self.colon_trivia = colon_trivia if colon_trivia is not None else []

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered SSA value string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += self.name
    if self.use_index is not None:
      out += f"#{self.use_index}"
    if self.type_node:
      out += "".join(t.text for t in self.colon_trivia)
      out += ":"
      out += self.type_node.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class AttributeNode(MlirNode):
  """Represent a named attribute."""

  name: str = ""
  use_index: Optional[int] = None
  value: Any = ""
  type_annotation: Optional[str] = None

  def __init__(
    self,
    name: str = "",
    value: Any = "",
    type_annotation: Optional[str] = None,
    use_index: Optional[int] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize AttributeNode.

    Args:
        name: Attribute name.
        value: Attribute value string or list of strings.
        type_annotation: Optional type annotation string.
        use_index: Optional index.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.value = value
    self.type_annotation = type_annotation
    self.use_index = use_index

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered attribute string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    suffix = f" : {self.type_annotation}" if self.type_annotation else ""

    if isinstance(self.value, list):
      val_str = f"[{', '.join(self.value)}]"
    else:
      val_str = str(self.value)

    out += f"{self.name} = {val_str}{suffix}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class BlockNode(MlirNode):
  """Represent a Basic Block within a Region."""

  label: str = ""
  operations: List["OperationNode"] = field(default_factory=list)
  arguments: List[Tuple[ValueNode, TypeNode]] = field(default_factory=list)

  def __init__(
    self,
    label: str = "",
    operations: Optional[List["OperationNode"]] = None,
    arguments: Optional[List[Tuple[ValueNode, TypeNode]]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize BlockNode.

    Args:
        label: Block label string.
        operations: List of OperationNode instances.
        arguments: List of (ValueNode, TypeNode) block arguments.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.label = label
    self.operations = operations if operations is not None else []
    self.arguments = arguments if arguments is not None else []

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered block string.
    """
    out = "".join(t.text for t in self.leading_trivia)

    arg_strs = []
    for val, typ in self.arguments:
      arg_strs.append(f"{val.to_text()}: {typ.to_text()}")

    args_text = ""
    if arg_strs:
      args_text = f"({', '.join(arg_strs)})"

    if self.label:
      out += f"{self.label}{args_text}:"
      if not self.operations and not self.trailing_trivia:
        out += "\n"

    for op in self.operations:
      out += op.to_text()

    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class RegionNode(MlirNode):
  """Represent a Region containing Blocks."""

  blocks: List[BlockNode] = field(default_factory=list)

  def __init__(
    self,
    blocks: Optional[List[BlockNode]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize RegionNode.

    Args:
        blocks: List of BlockNode instances.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.blocks = blocks if blocks is not None else []

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered region string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += "{"
    for b in self.blocks:
      out += b.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    out += "}"
    return out


@dataclass
class OperationNode(MlirNode):
  """Represent a specific MLIR Operation."""

  name: str = ""
  use_index: Optional[int] = None
  is_generic: bool = False
  location: Optional[str] = None
  properties: List["AttributeNode"] = field(default_factory=list)
  successors: List[str] = field(default_factory=list)
  results: List[ValueNode] = field(default_factory=list)
  operands: List[ValueNode] = field(default_factory=list)
  attributes: List[AttributeNode] = field(default_factory=list)
  regions: List[RegionNode] = field(default_factory=list)
  result_types: List[TypeNode] = field(default_factory=list)
  op_tail_str: str = ""
  op_tail_trivia: List[Trivia] = field(default_factory=list)
  name_trivia: Any = field(default_factory=list)
  has_parens: bool = True

  def __init__(
    self,
    name: str = "",
    use_index: Optional[int] = None,
    is_generic: bool = False,
    location: Optional[str] = None,
    properties: Optional[List["AttributeNode"]] = None,
    successors: Optional[List[str]] = None,
    results: Optional[List[ValueNode]] = None,
    operands: Optional[List[ValueNode]] = None,
    attributes: Optional[List[AttributeNode]] = None,
    regions: Optional[List[RegionNode]] = None,
    result_types: Optional[List[TypeNode]] = None,
    op_tail_str: str = "",
    op_tail_trivia: Optional[List[Trivia]] = None,
    name_trivia: Any = None,
    has_parens: bool = True,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize OperationNode.

    Args:
        name: Operation name.
        use_index: Optional index.
        is_generic: Generic operation syntax flag.
        location: Optional location string.
        properties: Operation properties list.
        successors: Successor block names.
        results: Result ValueNode list.
        operands: Operand ValueNode list.
        attributes: AttributeNode list.
        regions: RegionNode list.
        result_types: Result TypeNode list.
        op_tail_str: Operation tail string.
        op_tail_trivia: Operation tail trivia list.
        name_trivia: Trivia following operation name.
        has_parens: Whether operand parentheses are present.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    self.name = name
    self.use_index = use_index
    self.is_generic = is_generic
    self.location = location
    self.properties = properties if properties is not None else []
    self.successors = successors if successors is not None else []
    self.results = results if results is not None else []
    self.operands = operands if operands is not None else []
    self.attributes = attributes if attributes is not None else []
    self.regions = regions if regions is not None else []
    self.result_types = result_types if result_types is not None else []
    self.op_tail_str = op_tail_str
    self.op_tail_trivia = op_tail_trivia if op_tail_trivia is not None else []
    self.name_trivia = name_trivia
    self.has_parens = has_parens
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def __post_init__(self) -> None:
    """Initialize node after dataclass creation."""
    super().__post_init__()
    if isinstance(self.name_trivia, str):
      self.name_trivia = [Trivia(self.name_trivia)]
    elif self.name_trivia is None:
      self.name_trivia = []

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered operation string.
    """
    parts = []
    for t in self.leading_trivia:
      parts.append(t.text)

    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    if getattr(self, "is_generic", False):
      parts.append(f'"{self.name}"')
    else:
      parts.append(self.name)

    if self.name_trivia:
      for t in self.name_trivia:
        parts.append(t.text)

    if self.operands:
      if not self.name_trivia and not self.has_parens:
        # Check if first operand has leading trivia, if not, add space
        if not self.operands[0].leading_trivia:
          parts.append(" ")
      elif not self.name_trivia and self.has_parens:
        parts.append(" ")
      op_names = []
      for i, o in enumerate(self.operands):
        # If no trivia and not first operand, add space
        if i > 0 and not o.leading_trivia:
          op_names.append(" " + o.to_text())
        else:
          op_names.append(o.to_text())

      if self.has_parens:
        parts.append(f"({','.join(op_names)})")
      else:
        parts.append(",".join(op_names))

    if self.successors:
      parts.append(" [")
      parts.append(", ".join(self.successors))
      parts.append("]")

    if self.properties:
      parts.append(" <{")
      parts.append(", ".join([a.to_text() for a in self.properties]))
      parts.append("}>")

    if self.location:
      parts.append(f" loc({self.location})")

    if self.attributes:
      if not self.name_trivia:
        parts.append(" ")
      parts.append("{")
      attrs_str = ", ".join([a.to_text() for a in self.attributes])
      parts.append(attrs_str)
      parts.append("}")

    if self.result_types:
      for t in self.op_tail_trivia:
        parts.append(t.text)
      if not self.op_tail_str:
        if not self.op_tail_trivia and not self.name_trivia:
          parts.append(" : ")
        else:
          parts.append(": ")
      else:
        parts.append(self.op_tail_str)

      if len(self.result_types) == 1:
        parts.append(self.result_types[0].to_text())
      else:
        r_types = [t.to_text() for t in self.result_types]
        if self.op_tail_str == "-> ":
          parts.append(f"({', '.join(r_types)})")  # e.g. -> (tensor<...>, tensor<...>)
        elif not self.op_tail_str:
          parts.append(f"({', '.join(r_types)})")
        else:
          parts.append(", ".join(r_types))

    if self.regions:
      if parts and not parts[-1].endswith(" ") and not self.regions[0].leading_trivia:
        parts.append(" ")
      for reg in self.regions:
        parts.append(reg.to_text())

    for t in self.trailing_trivia:
      parts.append(t.text)

    out = "".join(parts)
    return out


@dataclass
class StableHloConstantOp(OperationNode):
  """Specialized node for stablehlo.constant preserving dialect trivia."""

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered constant operation string.
    """
    parts = []
    for t in self.leading_trivia:
      parts.append(t.text)

    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    parts.append(self.name)

    if self.name_trivia:
      for t in self.name_trivia:
        parts.append(t.text)

    if self.successors:
      parts.append(" [")
      parts.append(", ".join(self.successors))
      parts.append("]")

    if self.properties:
      parts.append(" <{")
      parts.append(", ".join([a.to_text() for a in self.properties]))
      parts.append("}>")

    if self.location:
      parts.append(f" loc({self.location})")

    if self.attributes:
      if not self.name_trivia:
        parts.append(" ")
      attr = self.attributes[0]
      parts.append(str(attr.value))

    if self.result_types:
      parts.append(" : ")
      if len(self.result_types) == 1:
        parts.append(self.result_types[0].to_text())
      else:
        t_names = [t.to_text() for t in self.result_types]
        parts.append(f"({', '.join(t_names)})")

    for t in self.trailing_trivia:
      parts.append(t.text)

    out = "".join(parts)
    return out


@dataclass
class AttributeAliasDefNode(MlirNode):
  """Represent a top-level attribute alias definition."""

  name: str = ""
  use_index: Optional[int] = None
  value_node: Optional[MlirNode] = None
  value_str: str = ""

  def __init__(
    self,
    name: str = "",
    use_index: Optional[int] = None,
    value_node: Optional[MlirNode] = None,
    value_str: str = "",
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize AttributeAliasDefNode.

    Args:
        name: Alias name.
        use_index: Optional index.
        value_node: Optional MlirNode value.
        value_str: Raw string value.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.use_index = use_index
    self.value_node = value_node
    self.value_str = value_str

  def to_text(self) -> str:
    """Return textual representation.

    Returns:
        str: The rendered attribute alias string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += f"{self.name} = "
    if self.value_node:
      out += self.value_node.to_text()
    else:
      out += self.value_str
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class TypeAliasDefNode(MlirNode):
  """Represent a type alias definition."""

  name: str = ""
  use_index: Optional[int] = None
  type_node: Optional[TypeNode] = None
  leading_trivia: List[Trivia] = field(default_factory=list)
  trailing_trivia: List[Trivia] = field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    type_node: Optional[TypeNode] = None,
    use_index: Optional[int] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize TypeAliasDefNode.

    Args:
        name: Type alias name.
        type_node: TypeNode definition.
        use_index: Optional index.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.name = name
    self.type_node = type_node
    self.use_index = use_index

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered type alias string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += f"!{self.name} ={self.type_node.to_text() if self.type_node else ''}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


@dataclass
class ModuleNode(MlirNode):
  """Top-level container."""

  body: BlockNode = field(default_factory=BlockNode)
  aliases: List[Any] = field(default_factory=list)

  def __init__(
    self,
    body: Optional[BlockNode] = None,
    aliases: Optional[List[Any]] = None,
    leading_trivia: Any = None,
    trailing_trivia: Any = None,
  ) -> None:
    """Initialize ModuleNode.

    Args:
        body: Module body BlockNode.
        aliases: List of top-level alias nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    super().__init__(
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )
    self.body = body if body is not None else BlockNode()
    self.aliases = aliases if aliases is not None else []

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered module string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    for alias in self.aliases:
      out += alias.to_text()
    out += self.body.to_text()
    out += "".join(t.text for t in self.trailing_trivia)
    return out


TriviaNode = Trivia

__all__ = [
  "AttributeAliasDefNode",
  "AttributeNode",
  "BlockNode",
  "MlirNode",
  "ModuleNode",
  "OperationNode",
  "RegionNode",
  "StableHloConstantOp",
  "Trivia",
  "TriviaNode",
  "TypeAliasDefNode",
  "TypeNode",
  "ValueNode",
]
