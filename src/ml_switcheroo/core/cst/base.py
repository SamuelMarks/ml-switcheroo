"""Core Concrete Syntax Tree (CST) definitions and visitor/transformer bases.

This module provides the fundamental building blocks for representing
and manipulating code as a Concrete Syntax Tree, completely in pure Python
for maximum compatibility with environments like Pyodide/WASM.
"""

import typing

import pydantic

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
    pydantic.BaseModel.__init__(self, text=content if content and not text else text, kind=kind)

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


class CSTNode(pydantic.BaseModel):
  """Execute base class for all nodes in the Concrete Syntax Tree.

  Attributes:
      leading_trivia (List[Trivia]): Trivia immediately preceding this node.
      trailing_trivia (List[Trivia]): Trivia immediately following this node.
  """

  model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  leading_trivia: list[Trivia] = pydantic.Field(default_factory=list)
  trailing_trivia: list[Trivia] = pydantic.Field(default_factory=list)

  def __setattr__(self, name: str, value: typing.Hashable | typing.Sized) -> None:
    """Set attribute on CSTNode, falling back to object.__setattr__ for dynamic attributes.

    Args:
        name: Attribute name.
        value: Attribute value.
    """
    try:
      super().__setattr__(name, value)
    except ValueError:
      object.__setattr__(self, name, value)

  @pydantic.field_validator("leading_trivia", "trailing_trivia", mode="before")
  @classmethod
  def _coerce_trivia(cls, v: list[Trivia] | str | None) -> list[Trivia]:
    """Coerce string or None trivia inputs into a list of Trivia instances.

    Args:
        v: Raw trivia value.

    Returns:
        Normalized list of Trivia instances.
    """
    if isinstance(v, str):
      return [Trivia(v)]
    if v is None:
      return []
    return list(v)

  @pydantic.model_validator(mode="after")
  def _run_post_init(self) -> "CSTNode":
    """Execute post-initialization hook on CSTNode and subclasses.

    Returns:
        CSTNode: The initialized CSTNode instance.
    """
    self.__post_init__()
    return self

  def __post_init__(self) -> None:
    """Post-initialization hook for CSTNode subclasses."""
    return None

  def to_text(self) -> str:
    """Convert the node back to its exact string representation.

    This method must be implemented by all subclasses.

    Returns:
        str: The source code string, including all trivia.
    """
    raise NotImplementedError("Subclasses must implement to_text()")

  def __str__(self) -> str:
    """Alia for to_text() to integrate easily with built-ins.

    Returns:
        str: The source code string.
    """
    return self.to_text()


class CSTVisitor:
  """Base class for traversing a CST without modifying it.

  This uses a visitor pattern, dynamically dispatching to methods
  based on the node's class name.
  """

  def visit(self, node: CSTNode) -> None:
    """Visit a node and all of its children.

    Args:
        node (CSTNode): The node to visit.
    """
    class_name = type(node).__name__
    visit_method = getattr(self, f"visit_{class_name}", self.generic_visit)
    visit_method(node)

  def generic_visit(self, node: CSTNode) -> None:
    """Execute default visitation logic if no specific method is found.

    Args:
        node (CSTNode): The node whose children should be visited.
    """
    for field_name in type(node).model_fields:
      value = getattr(node, field_name)
      if isinstance(value, CSTNode):
        self.visit(value)
      elif isinstance(value, list):
        for item in value:
          if isinstance(item, CSTNode):
            self.visit(item)


NodeT = typing.TypeVar("NodeT", bound=CSTNode)


class CSTTransformer:
  """Base class for transforming a CST.

  Methods can return a modified node or a completely new node.
  """

  def transform(self, node: NodeT) -> NodeT:
    """Transform a node and its children.

    Args:
        node (NodeT): The node to transform.

    Returns:
        NodeT: The transformed node.
    """
    class_name = type(node).__name__
    transform_method = getattr(self, f"transform_{class_name}", self.generic_transform)
    res: NodeT = transform_method(node)
    return res

  def generic_transform(self, node: NodeT) -> NodeT:
    """Execute default transformation logic if no specific method is found.

    Args:
        node (NodeT): The node whose children should be transformed.

    Returns:
        NodeT: The node, potentially modified.
    """
    for field_name in type(node).model_fields:
      value = getattr(node, field_name)
      if isinstance(value, CSTNode):
        setattr(node, field_name, self.transform(value))
      elif isinstance(value, list):
        new_list = []
        for item in value:
          if isinstance(item, CSTNode):
            new_list.append(self.transform(item))
          else:
            new_list.append(item)
        setattr(node, field_name, new_list)
    return node
