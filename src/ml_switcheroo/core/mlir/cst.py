"""MLIR Concrete Syntax Tree Nodes.

This module defines the data structures for representing MLIR source code.
It ensures structural hierarchy (Module -> Operation -> Region -> Block)
and trivia preservation for high-fidelity round-tripping.
"""

import typing

import pydantic
from pydantic import JsonValue

from ml_switcheroo.core.cst.base import CstFieldValue, CSTNode, Trivia
from ml_switcheroo.core.cst.base import TriviaInput as TriviaInput


class MlirNode(CSTNode):
  """Abstract base class for all MLIR CST nodes."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize MlirNode.

    Args:
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to CSTNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)


class TypeNode(MlirNode):
  """Represent a type annotation.

  Attributes:
      body: The body (`str`).
  """

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
    super().__init__(
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
  """Represent an SSA Value identifier (e.g. %0).

  Attributes:
      name: The name (`str`).
      use_index: The use index (`int | None`).
      type_node: The type node (`TypeNode | None`).
      colon_trivia: The colon trivia (`list[Trivia]`).
  """

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
    super().__init__(
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
  """Represent a named attribute.

  Attributes:
      name: The name (`str`).
      use_index: The use index (`int | None`).
      value: The value (`str | int | float | bool | list[str] | MlirNode | JsonValue`).
      type_annotation: The type annotation (`str | None`).
  """

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
    super().__init__(
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

    val_str = f"[{', '.join(str(x) for x in self.value)}]" if isinstance(self.value, list) else str(self.value)

    out += f"{self.name} = {val_str}{suffix}"
    out += "".join(t.text for t in self.trailing_trivia)
    return out


class BlockNode(MlirNode):
  """Represent a Basic Block within a Region.

  Attributes:
      label: The label (`str`).
      operations: The operations (`list['OperationNode']`).
      arguments: The arguments (`list[tuple[ValueNode, TypeNode]]`).
  """

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
    super().__init__(
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
  """Represent a Region containing Blocks.

  Attributes:
      blocks: The blocks (`list[BlockNode]`).
  """

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
    super().__init__(
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
  """Represent a specific MLIR Operation.

  Attributes:
      name: The name (`str`).
      use_index: The use index (`int | None`).
      is_generic: The is generic (`bool`).
      location: The location (`str | None`).
      properties: The properties (`list['AttributeNode']`).
      successors: The successors (`list[str]`).
      results: The results (`list[ValueNode]`).
      operands: The operands (`list[ValueNode]`).
      attributes: The attributes (`list[AttributeNode]`).
      regions: The regions (`list[RegionNode]`).
      result_types: The result types (`list[TypeNode]`).
      op_tail_str: The op tail str (`str`).
      op_tail_trivia: The op tail trivia (`list[Trivia]`).
      name_trivia: The name trivia (`str | list[Trivia] | None`).
      has_parens: The has parens (`bool`).
  """

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

  @pydantic.field_validator(
    "properties",
    "successors",
    "results",
    "operands",
    "attributes",
    "regions",
    "result_types",
    "op_tail_trivia",
    mode="before",
  )
  @classmethod
  def _coerce_optional_list(cls, v: CstFieldValue) -> CstFieldValue:
    """Normalize ``None`` list field inputs into empty lists during construction.

    Args:
        v: Raw list field input or ``None``.

    Returns:
        Empty list when ``v`` is ``None``, otherwise ``v`` unchanged.
    """
    return [] if v is None else v

  @pydantic.field_validator("name_trivia", mode="before")
  @classmethod
  def _coerce_name_trivia(cls, v: TriviaInput) -> list[Trivia]:
    """Coerce string or None name_trivia inputs into a list of Trivia instances.

    Args:
        v: Raw name_trivia value.

    Returns:
        Normalized list of Trivia instances.
    """
    return [Trivia(v)] if isinstance(v, str) else cls._coerce_trivia(v)

  def _post_init(self) -> None:
    """Initialize node after creation."""
    super()._post_init()

  def _format_header(self) -> str:
    """Render leading trivia, SSA results prefix, operation name, and name trivia.

    Returns:
        str: Formatted operation header string.
    """
    parts: list[str] = [t.text for t in self.leading_trivia]
    if self.results:
      parts.append(", ".join(r.to_text() for r in self.results))
      parts.append(" = ")
    parts.append(f'"{self.name}"' if self.is_generic else self.name)
    if isinstance(self.name_trivia, list) and self.name_trivia:
      parts.extend(t.text for t in self.name_trivia)
    return "".join(parts)

  def _format_operands(self) -> str:
    """Render SSA operands with leading spacing and optional parentheses.

    Returns:
        str: Formatted operands segment, or empty string when absent.
    """
    if not self.operands:
      return ""
    prefix = ""
    if not self.name_trivia and not self.has_parens:
      if not self.operands[0].leading_trivia:
        prefix = " "
    elif not self.name_trivia and self.has_parens:
      prefix = " "

    op_names = [
      (" " + o.to_text() if i > 0 and not o.leading_trivia else o.to_text()) for i, o in enumerate(self.operands)
    ]
    joined = ",".join(op_names)
    return f"{prefix}({joined})" if self.has_parens else f"{prefix}{joined}"

  def _format_metadata(self) -> str:
    """Render successors, properties, source location, and attribute dictionary.

    Returns:
        str: Formatted metadata segment.
    """
    parts: list[str] = []
    if self.successors:
      parts.append(f" [{', '.join(self.successors)}]")
    if self.properties:
      parts.append(f" <{{{', '.join(a.to_text() for a in self.properties)}}}>")
    if self.location:
      parts.append(f" loc({self.location})")
    if self.attributes:
      if not self.name_trivia:
        parts.append(" ")
      parts.append(f"{{{', '.join(a.to_text() for a in self.attributes)}}}")
    return "".join(parts)

  def _format_result_types(self) -> str:
    """Render operation tail trivia, separator, and result type annotations.

    Returns:
        str: Formatted result types segment, or empty string when absent.
    """
    if not self.result_types:
      return ""
    parts: list[str] = [t.text for t in self.op_tail_trivia]
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
      joined = ", ".join(t.to_text() for t in self.result_types)
      wrap_parens = self.op_tail_str == "-> " or not self.op_tail_str
      parts.append(f"({joined})" if wrap_parens else joined)
    return "".join(parts)

  def _format_regions(self, current_text: str) -> str:
    """Render attached regions with separating space when required.

    Args:
        current_text: Rendered prefix preceding the region list.

    Returns:
        str: Formatted regions segment, or empty string when absent.
    """
    if not self.regions:
      return ""
    prefix = " " if current_text and not current_text.endswith(" ") and not self.regions[0].leading_trivia else ""
    return prefix + "".join(reg.to_text() for reg in self.regions)

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered operation string.
    """
    body = self._format_header() + self._format_operands() + self._format_metadata() + self._format_result_types()
    regions_str = self._format_regions(body)
    trailing = "".join(t.text for t in self.trailing_trivia)
    return body + regions_str + trailing


class StableHloConstantOp(OperationNode):
  """Specialized node for stablehlo.constant preserving dialect trivia."""

  def to_text(self) -> str:
    """Return textual representation of the node.

    Returns:
        str: The rendered constant operation string.
    """
    parts: list[str] = [t.text for t in self.leading_trivia]

    if self.results:
      r_names = [r.to_text() for r in self.results]
      parts.append(", ".join(r_names))
      parts.append(" = ")

    parts.append(self.name)

    if isinstance(self.name_trivia, list) and self.name_trivia:
      parts.extend(t.text for t in self.name_trivia)

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

    parts.extend(t.text for t in self.trailing_trivia)

    return "".join(parts)


class AttributeAliasDefNode(MlirNode):
  """Represent a top-level attribute alias definition.

  Attributes:
      name: The name (`str`).
      use_index: The use index (`int | None`).
      value_node: The value node (`MlirNode | None`).
      value_str: The value str (`str`).
  """

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
    super().__init__(
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
  """Represent a type alias definition.

  Attributes:
      name: The name (`str`).
      use_index: The use index (`int | None`).
      type_node: The type node (`TypeNode | None`).
      leading_trivia: The leading trivia (`list[Trivia]`).
      trailing_trivia: The trailing trivia (`list[Trivia]`).
  """

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
    super().__init__(
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
  """Top-level container.

  Attributes:
      body: The body (`BlockNode`).
      aliases: The aliases (`list[AliasNodeUnion]`).
  """

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
    super().__init__(
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
