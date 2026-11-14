"""Core Concrete Syntax Tree (CST) definitions and visitor/transformer bases.

This module provides the fundamental building blocks for representing
and manipulating code as a Concrete Syntax Tree, completely in pure Python
for maximum compatibility with environments like Pyodide/WASM.
"""

import enum
import typing

import pydantic
from pydantic import JsonValue

from ml_switcheroo.exceptions import CSTValidationError

T = typing.TypeVar("T")


class Trivia(pydantic.BaseModel):
  """Represent non-semantic tokens, such as whitespace and comments.

  Attributes:
      text (str): The literal string content of the trivia (e.g., "   " or "// comment").
      kind (str): Optional classification of trivia ("whitespace", "comment", "newline").
  """

  model_config = pydantic.ConfigDict(extra="forbid")

  text: str = ""
  kind: str = "whitespace"

  def __init__(
    self,
    text: str = "",
    kind: str = "whitespace",
    *,
    content: str = "",
  ) -> None:
    """Initialize Trivia instance.

    Args:
        text: The literal string content of the trivia.
        kind: Optional classification of trivia.
        content: Optional alias for ``text``.
    """
    super().__init__(text=content if content and not text else text, kind=kind)

  @property
  def content(self) -> str:
    """Return literal trivia text content.

    Returns:
        str: The literal trivia string.
    """
    return self.text

  def to_text(self) -> str:
    """Return the raw text of the trivia.

    Returns:
        str: The literal trivia string.
    """
    return self.text

  def emit(self, indent_level: int = 0) -> str:
    """Emit the raw text of the trivia.

    Args:
        indent_level: Indentation depth (unused for literal trivia).

    Returns:
        str: The literal trivia string.
    """
    return self.text


@typing.runtime_checkable
class _TriviaLikeProtocol(typing.Protocol):
  """Protocol for markup or dialect trivia objects exposing ``content`` and ``kind``.

  Attributes:
      content: Literal trivia string content.
      kind: Trivia category identifier.
  """

  content: str
  kind: str


TriviaInput: typing.TypeAlias = typing.Sequence[Trivia | _TriviaLikeProtocol | str] | str | None

CstFieldValue: typing.TypeAlias = (
  pydantic.BaseModel
  | TriviaInput
  | JsonValue
  | enum.Enum
  | typing.Sequence["CstFieldValue"]
  | typing.Mapping[str, "CstFieldValue"]
)


class CSTNode(pydantic.BaseModel):
  """Execute base class for all nodes in the Concrete Syntax Tree.

  Attributes:
      leading_trivia (List[Trivia]): Trivia immediately preceding this node.
      trailing_trivia (List[Trivia]): Trivia immediately following this node.
  """

  model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  leading_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  trailing_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize CSTNode.

    Args:
        leading_trivia: Optional leading trivia.
        trailing_trivia: Optional trailing trivia.
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(
      leading_trivia=self._coerce_trivia(leading_trivia),
      trailing_trivia=self._coerce_trivia(trailing_trivia),
      **kwargs,
    )

  @pydantic.field_validator("leading_trivia", "trailing_trivia", mode="before")
  @classmethod
  def _coerce_trivia(cls, v: TriviaInput) -> list[Trivia]:
    """Coerce string, ``None``, or trivia-like sequence inputs into a list of ``Trivia`` instances.

    Args:
        v: Raw trivia value.

    Returns:
        Normalized list of Trivia instances.
    """
    if v is None:
      return []
    if isinstance(v, str):
      return [Trivia(v)] if v else []
    out: list[Trivia] = []
    for item in v:
      if isinstance(item, Trivia):
        out.append(item)
      elif isinstance(item, _TriviaLikeProtocol):
        out.append(Trivia(text=item.content, kind=item.kind))
      else:
        out.append(Trivia(str(item)))
    return out

  def leading_trivia_text(self) -> str:
    """Concatenate all leading trivia items into a single string.

    Returns:
        str: Concatenated leading trivia text.
    """
    return "".join(t.text for t in self.leading_trivia)

  def trailing_trivia_text(self) -> str:
    """Concatenate all trailing trivia items into a single string.

    Returns:
        str: Concatenated trailing trivia text.
    """
    return "".join(t.text for t in self.trailing_trivia)

  @pydantic.model_validator(mode="after")
  def _run_post_init(self) -> "CSTNode":
    """Execute post-initialization hook on CSTNode and subclasses.

    Returns:
        CSTNode: The initialized CSTNode instance.
    """
    self._post_init()
    return self

  def _post_init(self) -> None:
    """Validate or normalize node state after Pydantic initialization."""
    return

  def to_text(self) -> str:
    """Convert the node back to its exact string representation.

    This method must be implemented by all subclasses.

    Returns:
        str: The source code string, including all trivia.

    Raises:
        NotImplementedError: Must be implemented by concrete `CSTNode` subclasses.
    """
    raise NotImplementedError("Subclasses must implement to_text()")

  def emit(self, indent_level: int = 0) -> str:
    """Emit the node's string representation, optionally indented by ``indent_level``.

    Args:
        indent_level: Indentation depth (two spaces per level when positive).

    Returns:
        str: Emitted source text representation.
    """
    text = self.to_text()
    if indent_level <= 0:
      return text
    prefix = "  " * indent_level
    return "\n".join(f"{prefix}{line}" if line else line for line in text.split("\n"))

  def __str__(self) -> str:
    """Alias for to_text() to integrate easily with built-ins.

    Returns:
        str: The source code string.
    """
    return self.to_text()


NodeT = typing.TypeVar("NodeT", bound=CSTNode)


class CSTVisitor:
  """Base class for traversing a CST via explicit type-keyed handler registration."""

  def __init__(self) -> None:
    """Initialize the type-keyed visitor dispatch table."""
    self._visitors: dict[type[CSTNode], typing.Callable[[CSTNode], None]] = {}

  def register_visitor(
    self,
    node_type: type[NodeT],
    handler: typing.Callable[[NodeT], None],
  ) -> None:
    """Register a typed visitor callback for ``node_type``.

    Args:
        node_type: Concrete ``CSTNode`` subclass to match.
        handler: Callback invoked when visiting an instance of ``node_type``.
    """

    def _wrapped(node: CSTNode) -> None:
      """Narrow ``node`` to ``node_type`` and invoke ``handler``.

      Args:
          node: Visited CST node matching ``node_type``.

      Raises:
          CSTValidationError: If ``node`` is not an instance of ``node_type``.
      """
      if not isinstance(node, node_type):
        raise CSTValidationError(f"Expected {node_type.__name__}, got {type(node).__name__}.")
      handler(node)

    self._visitors[node_type] = _wrapped

  def visit(self, node: CSTNode) -> None:
    """Visit ``node`` using its registered handler or fall back to ``generic_visit``.

    Args:
        node (CSTNode): The node to visit.
    """
    handler = self._visitors.get(type(node))
    if handler is None:
      self.generic_visit(node)
      return
    handler(node)

  def generic_visit(self, node: CSTNode) -> None:
    """Execute default visitation logic across scalar and sequence ``CSTNode`` fields.

    Args:
        node (CSTNode): The node whose children should be visited.
    """
    for _, value in node:
      if isinstance(value, CSTNode):
        self.visit(value)
      elif isinstance(value, list):
        for item in value:
          if isinstance(item, CSTNode):
            self.visit(item)


class CSTTransformer:
  """Base class for transforming a CST via explicit type-keyed handler registration."""

  def __init__(self) -> None:
    """Initialize the type-keyed transformer dispatch table."""
    self._transformers: dict[type[CSTNode], typing.Callable[[CSTNode], CSTNode]] = {}

  def register_transformer(
    self,
    node_type: type[NodeT],
    handler: typing.Callable[[NodeT], CSTNode],
  ) -> None:
    """Register a typed transformation callback for ``node_type``.

    Args:
        node_type: Concrete ``CSTNode`` subclass to match.
        handler: Callback returning a replacement ``CSTNode``.
    """

    def _wrapped(node: CSTNode) -> CSTNode:
      """Narrow ``node`` to ``node_type`` and invoke ``handler``.

      Args:
          node: CST node matching ``node_type``.

      Returns:
          Transformed CST node.

      Raises:
          CSTValidationError: If ``node`` is not an instance of ``node_type``.
      """
      if not isinstance(node, node_type):
        raise CSTValidationError(f"Expected {node_type.__name__}, got {type(node).__name__}.")
      return handler(node)

    self._transformers[node_type] = _wrapped

  def transform(self, node: NodeT) -> NodeT:
    """Transform ``node`` via its registered handler or fall back to ``generic_transform``.

    Args:
        node (NodeT): The node to transform.

    Returns:
        NodeT: The transformed node.

    Raises:
        CSTValidationError: If a registered handler returns an incompatible node type.
    """
    handler = self._transformers.get(type(node))
    if handler is None:
      return self.generic_transform(node)
    res = handler(node)
    if not isinstance(res, type(node)):
      raise CSTValidationError(
        f"Transformer for '{type(node).__name__}' must return {type(node).__name__}, got {type(res).__name__}"
      )
    return res

  def generic_transform(self, node: NodeT) -> NodeT:
    """Execute default child transformation logic when no specific handler is registered.

    Args:
        node (NodeT): The node whose children should be transformed.

    Returns:
        NodeT: A copy of `node` with all scalar and sequence `CSTNode` children transformed.
    """
    updates: dict[str, CstFieldValue] = {}
    for field_name, value in node:
      if isinstance(value, CSTNode):
        updates[field_name] = self.transform(value)
      elif isinstance(value, list) and any(isinstance(item, CSTNode) for item in value):
        updates[field_name] = [self.transform(item) if isinstance(item, CSTNode) else item for item in value]
    if not updates:
      return node
    return node.model_copy(update=updates)
