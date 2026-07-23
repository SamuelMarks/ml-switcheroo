"""Function Signature Extraction using LibCST.

This module provides a robust, CST-based utility to safely extract function
definitions and signatures from raw Python source code, replacing fragile
regex-based implementations.
"""

from typing import Optional

import libcst as cst


class FunctionDefVisitor(cst.CSTVisitor):
  """Visits CST to find the first FunctionDef node."""

  def __init__(self) -> None:
    """Initialize the visitor."""
    super().__init__()
    self.function_name: Optional[str] = None

  def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
    """Extract the name of the first function definition encountered.

    Args:
        node: The FunctionDef node.

    Returns:
        False to stop traversing (we only want the first function).
    """
    if self.function_name is None:
      self.function_name = node.name.value
      return False  # Stop visiting children
    return False


class SignatureExtractor:
  """Utility class to extract signatures from Python code."""

  @staticmethod
  def extract_first_function_name(code: str) -> Optional[str]:
    """Extracts the name of the first function defined in the source code.

    Args:
        code: The Python source code string.

    Returns:
        The name of the first function if found, else None.
    """
    try:
      module = cst.parse_module(code)
    except Exception:
      # Handle invalid syntax gracefully, returning None
      return None

    visitor = FunctionDefVisitor()
    module.visit(visitor)
    return visitor.function_name
