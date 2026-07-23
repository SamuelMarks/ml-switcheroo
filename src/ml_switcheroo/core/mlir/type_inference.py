"""Static Type Inference Engine for MLIR Emitters."""

from typing import Dict, Optional, List
import libcst as cst
from ml_switcheroo.core.mlir.types import (
  MLIRType,
  IntegerType,
  FloatType,
  TensorType,
)


def parse_py_type_to_mlir(type_str: str) -> MLIRType:
  """Parses a Python type annotation string into an MLIRType.

  Supports basic forms like 'int', 'float', 'bool', and naive tensor parsing
  like 'Tensor[f32]' or fallback to unranked 'tensor<*xf32>'.

  Args:
      type_str: Python Type Hint string.

  Returns:
      An MLIRType representation.

  """
  clean = type_str.lower().strip()
  if clean == "int":
    return IntegerType(32)
  if clean == "float":
    return FloatType("f32")
  if clean == "bool":
    return IntegerType(1)

  if "tensor" in clean or "array" in clean:
    # A generous default unranked tensor of floats
    return TensorType(FloatType("f32"), None)

  # Unknown fallback
  return FloatType("!sw.unknown")


class TypeInferencePass(cst.CSTVisitor):
  """A naive static type inference pass over a CST node.

  Computes intermediate tensor shapes and data types based on a provided
  environment of known types.
  """

  def __init__(self, initial_env: Optional[Dict[str, MLIRType]] = None):
    """Initialize the type inference pass.

    Args:
        initial_env: Initial mapping of variable names to their MLIRType.
    """
    self.env: Dict[str, MLIRType] = initial_env or {}
    self.return_types: List[MLIRType] = []

  def visit_Assign(self, node: cst.Assign) -> None:
    """Infer types for variable assignments.

    Args:
        node: The Assign node.
    """
    # For now, default to unranked f32 tensor if we don't know
    inferred_type = self._infer_expression(node.value)

    for target in node.targets:
      t = target.target
      if isinstance(t, cst.Name):  # pragma: no branch
        self.env[t.value] = inferred_type

  def visit_Return(self, node: cst.Return) -> None:
    """Capture the return type.

    Args:
        node: The Return node.
    """
    if node.value:
      self.return_types.append(self._infer_expression(node.value))
    else:
      # Void return? Just store empty
      pass

  def _infer_expression(self, node: cst.BaseExpression) -> MLIRType:
    """Helper to infer the type of an expression.

    Args:
        node: The expression node.

    Returns:
        The inferred MLIRType.
    """
    if isinstance(node, cst.Name):
      return self.env.get(node.value, TensorType(FloatType("f32"), None))

    if isinstance(node, cst.Float):
      return FloatType("f32")
    if isinstance(node, cst.Integer):
      return IntegerType(32)

    # Default to unranked f32 tensor for unknown operations
    return TensorType(FloatType("f32"), None)
