"""MLIR Concrete Syntax Tree Nodes.

This module defines the data structures for representing MLIR source code.
It ensures structural hierarchy (Module -> Operation -> Region -> Block)
and trivia preservation for high-fidelity round-tripping.
"""

import typing

import pydantic
from pydantic import JsonValue

from ml_switcheroo.core.cst.base import CSTNode, Trivia

TriviaInput: typing.TypeAlias = list[Trivia] | str | None


class MlirNode(CSTNode):
  """Abstract base class for all MLIR CST nodes."""


class TypeNode(MlirNode):
  """Represent a type annotation."""

  body: str = ""

  def __init__(
    self,
    body: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TypeNode.

    Args:
        body: Type string body.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      body=body,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Return textual representation.

    Returns:
        str: The rendered type string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += self.body
    out += "".join(t.text for t in self.trailing_trivia)
    return out


class ValueNode(MlirNode):
  """Represent an SSA Value identifier (e.g. %0)."""

  name: str = ""
  use_index: int | None = None
  type_node: TypeNode | None = None
  colon_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    use_index: int | None = None,
    type_node: TypeNode | None = None,
    colon_trivia: list[Trivia] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    pydantic.BaseModel.__init__(
      self,
      name=name,
      use_index=use_index,
      type_node=type_node,
      colon_trivia=colon_trivia if colon_trivia is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class AttributeNode(MlirNode):
  """Represent a named attribute."""

  name: str = ""
  use_index: int | None = None
  value: str | int | float | bool | list[str] | MlirNode | JsonValue = ""
  type_annotation: str | None = None

  def __init__(
    self,
    name: str = "",
    value: str | int | float | bool | list[str] | MlirNode | JsonValue = "",
    type_annotation: str | None = None,
    use_index: int | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    pydantic.BaseModel.__init__(
      self,
      name=name,
      value=value,
      type_annotation=type_annotation,
      use_index=use_index,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered attribute string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    suffix = f" : {self.type_annotation}" if self.type_annotation else ""

    if isinstance(self.value, list):
      val_str = f"[{', '.join(str(x) for x in self.value)}]"
    else:
      val_str = str(self.value)

    out += f"{self.name} = {val_str}{suffix}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


class BlockNode(MlirNode):
  """Represent a Basic Block within a Region."""

  label: str = ""
  operations: list["OperationNode"] = pydantic.Field(default_factory=list)
  arguments: list[tuple[ValueNode, TypeNode]] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    label: str = "",
    operations: list["OperationNode"] | None = None,
    arguments: list[tuple[ValueNode, TypeNode]] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize BlockNode.

    Args:
        label: Block label string.
        operations: List of OperationNode instances.
        arguments: List of (ValueNode, TypeNode) block arguments.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      label=label,
      operations=operations if operations is not None else [],
      arguments=arguments if arguments is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class RegionNode(MlirNode):
  """Represent a Region containing Blocks."""

  blocks: list[BlockNode] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    blocks: list[BlockNode] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize RegionNode.

    Args:
        blocks: List of BlockNode instances.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      blocks=blocks if blocks is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class OperationNode(MlirNode):
  """Represent a specific MLIR Operation."""

  name: str = ""
  use_index: int | None = None
  is_generic: bool = False
  location: str | None = None
  properties: list["AttributeNode"] = pydantic.Field(default_factory=list)
  successors: list[str] = pydantic.Field(default_factory=list)
  results: list[ValueNode] = pydantic.Field(default_factory=list)
  operands: list[ValueNode] = pydantic.Field(default_factory=list)
  attributes: list[AttributeNode] = pydantic.Field(default_factory=list)
  regions: list[RegionNode] = pydantic.Field(default_factory=list)
  result_types: list[TypeNode] = pydantic.Field(default_factory=list)
  op_tail_str: str = ""
  op_tail_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  name_trivia: str | list[Trivia] | None = pydantic.Field(default_factory=list)
  has_parens: bool = True

  @pydantic.field_validator("name_trivia", mode="before")
  @classmethod
  def _coerce_name_trivia(cls, v: TriviaInput) -> list[Trivia]:
    """Coerce string or None name_trivia inputs into a list of Trivia instances.

    Args:
        v: Raw name_trivia value.

    Returns:
        Normalized list of Trivia instances.
    """
    return [Trivia(v)] if isinstance(v, str) else list(v or [])

  def __init__(
    self,
    name: str = "",
    use_index: int | None = None,
    is_generic: bool = False,
    location: str | None = None,
    properties: list["AttributeNode"] | None = None,
    successors: list[str] | None = None,
    results: list[ValueNode] | None = None,
    operands: list[ValueNode] | None = None,
    attributes: list[AttributeNode] | None = None,
    regions: list[RegionNode] | None = None,
    result_types: list[TypeNode] | None = None,
    op_tail_str: str = "",
    op_tail_trivia: list[Trivia] | None = None,
    name_trivia: TriviaInput = None,
    has_parens: bool = True,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    pydantic.BaseModel.__init__(
      self,
      name=name,
      use_index=use_index,
      is_generic=is_generic,
      location=location,
      properties=properties if properties is not None else [],
      successors=successors if successors is not None else [],
      results=results if results is not None else [],
      operands=operands if operands is not None else [],
      attributes=attributes if attributes is not None else [],
      regions=regions if regions is not None else [],
      result_types=result_types if result_types is not None else [],
      op_tail_str=op_tail_str,
      op_tail_trivia=op_tail_trivia if op_tail_trivia is not None else [],
      name_trivia=name_trivia if name_trivia is not None else [],
      has_parens=has_parens,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def __post_init__(self) -> None:
    """Initialize node after creation."""
    super().__post_init__()

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

    if isinstance(self.name_trivia, list) and self.name_trivia:
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

    if isinstance(self.name_trivia, list) and self.name_trivia:
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


class AttributeAliasDefNode(MlirNode):
  """Represent a top-level attribute alias definition."""

  name: str = ""
  use_index: int | None = None
  value_node: MlirNode | None = None
  value_str: str = ""

  def __init__(
    self,
    name: str = "",
    use_index: int | None = None,
    value_node: MlirNode | None = None,
    value_str: str = "",
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
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
    pydantic.BaseModel.__init__(
      self,
      name=name,
      use_index=use_index,
      value_node=value_node,
      value_str=value_str,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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


class TypeAliasDefNode(MlirNode):
  """Represent a type alias definition."""

  name: str = ""
  use_index: int | None = None
  type_node: TypeNode | None = None
  leading_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  trailing_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    name: str = "",
    type_node: TypeNode | None = None,
    use_index: int | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize TypeAliasDefNode.

    Args:
        name: Type alias name.
        type_node: TypeNode definition.
        use_index: Optional index.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      type_node=type_node,
      use_index=use_index,
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered type alias string.
    """
    out = "".join(t.text for t in self.leading_trivia)
    out += f"!{self.name} ={self.type_node.to_text() if self.type_node else ''}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


AliasNodeUnion: typing.TypeAlias = AttributeAliasDefNode | TypeAliasDefNode | MlirNode


class ModuleNode(MlirNode):
  """Top-level container."""

  body: BlockNode = pydantic.Field(default_factory=BlockNode)
  aliases: list[AliasNodeUnion] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    body: BlockNode | None = None,
    aliases: list[AliasNodeUnion] | None = None,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
  ) -> None:
    """Initialize ModuleNode.

    Args:
        body: Module body BlockNode.
        aliases: List of top-level alias nodes.
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
    """
    pydantic.BaseModel.__init__(
      self,
      body=body if body is not None else BlockNode(),
      aliases=aliases if aliases is not None else [],
      leading_trivia=leading_trivia if leading_trivia is not None else [],
      trailing_trivia=trailing_trivia if trailing_trivia is not None else [],
    )

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

BlockNode.model_rebuild()
ModuleNode.model_rebuild()

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
