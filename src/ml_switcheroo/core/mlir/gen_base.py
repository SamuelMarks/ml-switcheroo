"""MLIR Generator Base Utilities.

Defines the mixin with common helper functions used by Expression and Statement generators.
"""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import OperationNode


class BaseGeneratorMixin:
  """Base class providing common string manipulation and attribute retrieval helpers for code generation."""

  def _get_attr(self, op: OperationNode, key: str) -> typing.Optional[str]:
    """Retrieve the value of a specific attribute from an OperationNode.

    Args:
        op: The operation node to inspect.
        key: The attribute name key.

    Returns:
        The attribute value as string, or None if not found.

    """
    for attr in op.attributes:
      if attr.name == key:
        if isinstance(attr.value, list):
          return f"[{', '.join(str(x) for x in attr.value)}]"
        return str(attr.value)
    return None

  def _create_dotted_name(self, path: str) -> typing.Union[cst.Name, cst.Attribute]:
    """Create a LibCST Name/Attribute chain from a dot-separated string.

    Args:
        path: The python path string (e.g. "torch.nn.Conv2d").

    Returns:
        A CST expression node representing the name.

    """
    if not path:
      return cst.Name("unknown")
    parts = path.split(".")
    node: typing.Union[cst.Name, cst.Attribute] = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node
