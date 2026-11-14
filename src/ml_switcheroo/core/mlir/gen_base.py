"""MLIR Generator Common Utilities.

Provides ``MlirGenHelpers``, stateless helper utilities shared by
``MlirExpressionGenerator``, ``MlirStatementGenerator``, and ``MlirToPythonGenerator``.
"""

import libcst as cst

from ml_switcheroo.core.mlir.cst import OperationNode


class MlirGenHelpers:
  """Stateless helper utilities for extracting MLIR attributes and building dotted CST names."""

  @staticmethod
  def get_attr(op: OperationNode, key: str) -> str | None:
    """Retrieve the string representation of attribute ``key`` on ``op``.

    Args:
        op: MLIR ``OperationNode`` to inspect.
        key: Attribute name key.

    Returns:
        Attribute value string, or ``None`` if absent.
    """
    for attr in op.attributes:
      if attr.name == key:
        if isinstance(attr.value, list):
          return f"[{', '.join(str(x) for x in attr.value)}]"
        return str(attr.value)
    return None

  @staticmethod
  def create_dotted_name(path: str) -> cst.Name | cst.Attribute:
    """Construct a LibCST ``Name`` or ``Attribute`` chain from a dot-separated path.

    Args:
        path: Dotted Python identifier path (e.g. ``'torch.nn.Conv2d'``).

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute`` expression.
    """
    if not path:
      return cst.Name("unknown")
    parts = path.split(".")
    node: cst.Name | cst.Attribute = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node


__all__ = ["MlirGenHelpers"]
