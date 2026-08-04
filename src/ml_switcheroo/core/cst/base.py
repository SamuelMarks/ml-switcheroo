"""Core Concrete Syntax Tree (CST) definitions and visitor/transformer bases.

This module provides the fundamental building blocks for representing
and manipulating code as a Concrete Syntax Tree, completely in pure Python
for maximum compatibility with environments like Pyodide/WASM.
"""

from dataclasses import dataclass, field, fields
from typing import List, TypeVar

T = TypeVar("T")


@dataclass
class Trivia:
  """Represents non-semantic tokens, such as whitespace and comments.

  Attributes:
      text (str): The literal string content of the trivia (e.g., "   " or "// comment").
  """

  text: str


@dataclass
class CSTNode:
  """The base class for all nodes in the Concrete Syntax Tree.

  Attributes:
      leading_trivia (List[Trivia]): Trivia immediately preceding this node.
      trailing_trivia (List[Trivia]): Trivia immediately following this node.
  """

  leading_trivia: List[Trivia] = field(default_factory=list)
  trailing_trivia: List[Trivia] = field(default_factory=list)

  def __post_init__(self) -> None:
    """Ensures trivia is properly typed."""
    if isinstance(self.leading_trivia, str):
      self.leading_trivia = [Trivia(self.leading_trivia)]
    elif self.leading_trivia is None:
      self.leading_trivia = []
    if isinstance(self.trailing_trivia, str):
      self.trailing_trivia = [Trivia(self.trailing_trivia)]
    elif self.trailing_trivia is None:
      self.trailing_trivia = []

  def to_text(self) -> str:
    """Converts the node back to its exact string representation.

    This method must be implemented by all subclasses.

    Returns:
        str: The source code string, including all trivia.
    """
    raise NotImplementedError("Subclasses must implement to_text()")

  def __str__(self) -> str:
    """Alias for to_text() to integrate easily with built-ins."""
    return self.to_text()


class CSTVisitor:
  """Base class for traversing a CST without modifying it.

  This uses a visitor pattern, dynamically dispatching to methods
  based on the node's class name.
  """

  def visit(self, node: CSTNode) -> None:
    """Visits a node and all of its children.

    Args:
        node (CSTNode): The node to visit.
    """
    class_name = type(node).__name__
    visit_method = getattr(self, f"visit_{class_name}", self.generic_visit)
    visit_method(node)

  def generic_visit(self, node: CSTNode) -> None:
    """The default visitation logic if no specific method is found.

    Args:
        node (CSTNode): The node whose children should be visited.
    """
    for f in fields(node):
      value = getattr(node, f.name)
      if isinstance(value, CSTNode):
        self.visit(value)
      elif isinstance(value, list):
        for item in value:
          if isinstance(item, CSTNode):
            self.visit(item)


class CSTTransformer:
  """Base class for transforming a CST.

  Methods can return a modified node or a completely new node.
  """

  def transform(self, node: CSTNode) -> CSTNode:
    """Transforms a node and its children.

    Args:
        node (CSTNode): The node to transform.

    Returns:
        CSTNode: The transformed node.
    """
    class_name = type(node).__name__
    transform_method = getattr(self, f"transform_{class_name}", self.generic_transform)
    return transform_method(node)

  def generic_transform(self, node: CSTNode) -> CSTNode:
    """The default transformation logic if no specific method is found.

    Args:
        node (CSTNode): The node whose children should be transformed.

    Returns:
        CSTNode: The node, potentially modified.
    """
    for f in fields(node):
      value = getattr(node, f.name)
      if isinstance(value, CSTNode):
        setattr(node, f.name, self.transform(value))
      elif isinstance(value, list):
        new_list = []
        for item in value:
          if isinstance(item, CSTNode):
            new_list.append(self.transform(item))
          else:
            new_list.append(item)
        setattr(node, f.name, new_list)
    return node
