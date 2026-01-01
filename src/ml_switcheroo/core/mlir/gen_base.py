"""
MLIR Generator Base Utilities.

Defines the mixin with common helper functions used by Expression and Statement generators.
"""

from typing import Optional
import libcst as cst
from ml_switcheroo.core.mlir.nodes import OperationNode


class BaseGeneratorMixin:
  """
  Base class providing common string manipulation and attribute retrieval helpers for code generation.
  """

  def _get_attr(self, op: OperationNode, key: str) -> Optional[str]:
    """
    Retrieves the value of a specific attribute from an OperationNode.

    Args:
        op: The operation node to inspect.
        key: The attribute name key.

    Returns:
        The attribute value as string, or None if not found.
    """
    for attr in op.attributes:
      if attr.name == key:
        if isinstance(attr.value, list):
          return f"[{', '.join(attr.value)}]"
        return attr.value
    return None

  def _create_dotted_name(self, path: str) -> cst.BaseExpression:
    """
    Creates a LibCST Name/Attribute chain from a dot-separated string.

    Args:
        path: The python path string (e.g. "torch.nn.Conv2d").

    Returns:
        A CST expression node representing the name.
    """
    parts = path.split(".")
    if not parts:
      return cst.Name("unknown")
    node: cst.BaseExpression = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node
