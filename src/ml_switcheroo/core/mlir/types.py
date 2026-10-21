"""MLIR Type System.

This module defines strong typing structures for MLIR types, specifically focusing
on the types required for StableHLO operations (Tensors, Elements, etc.).
"""

from abc import ABC, abstractmethod

import pydantic


class MLIRType(pydantic.BaseModel, ABC):
  """Abstract base class for all strict MLIR types."""

  model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

  @abstractmethod
  def to_string(self) -> str:
    """Return the MLIR string representation of the type.

    Returns:
        str: The MLIR type string.
    """


class IntegerType(MLIRType):
  """Represent an MLIR Integer Type (e.g., i1, i32, i64)."""

  width: int

  def __init__(self, width: int) -> None:
    """Initialize IntegerType.

    Args:
        width: Bit width of the integer type.
    """
    pydantic.BaseModel.__init__(self, width=width)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'i32').

    Returns:
        str: The integer type string.
    """
    return f"i{self.width}"


class FloatType(MLIRType):
  """Represent an MLIR Float Type (e.g., f32, f64, bf16)."""

  name: str

  def __init__(self, name: str) -> None:
    """Initialize FloatType.

    Args:
        name: MLIR float type name.
    """
    pydantic.BaseModel.__init__(self, name=name)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'f32').

    Returns:
        str: The float type string.
    """
    return self.name


class ComplexType(MLIRType):
  """Represent an MLIR Complex Type (e.g., complex<f32>)."""

  element_type: FloatType

  def __init__(self, element_type: FloatType) -> None:
    """Initialize ComplexType.

    Args:
        element_type: Element FloatType of the complex type.
    """
    pydantic.BaseModel.__init__(self, element_type=element_type)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'complex<f32>').

    Returns:
        str: The complex type string.
    """
    return f"complex<{self.element_type.to_string()}>"


class TensorType(MLIRType):
  """Represent an MLIR Tensor Type (e.g., tensor<2x?xi32>)."""

  element_type: MLIRType
  shape: list[int | str] | None = None  # None means unranked (*), '?' means dynamic dimension

  def __init__(
    self,
    element_type: MLIRType,
    shape: list[int | str] | None = None,
  ) -> None:
    """Initialize TensorType.

    Args:
        element_type: Element MLIRType.
        shape: Optional list of dimension sizes or symbols.
    """
    pydantic.BaseModel.__init__(self, element_type=element_type, shape=shape)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'tensor<2x?xf32>').

    Returns:
        str: The tensor type string.
    """
    if self.shape is None:
      return f"tensor<*x{self.element_type.to_string()}>"

    if len(self.shape) == 0:
      return f"tensor<{self.element_type.to_string()}>"

    shape_strs = [str(dim) for dim in self.shape]
    shape_prefix = "x".join(shape_strs)
    return f"tensor<{shape_prefix}x{self.element_type.to_string()}>"


class FunctionType(MLIRType):
  """Represent an MLIR Function Type (e.g., (tensor<f32>) -> tensor<f32>)."""

  inputs: list[MLIRType]
  results: list[MLIRType]

  def __init__(self, inputs: list[MLIRType], results: list[MLIRType]) -> None:
    """Initialize FunctionType.

    Args:
        inputs: List of input MLIRType instances.
        results: List of result MLIRType instances.
    """
    pydantic.BaseModel.__init__(self, inputs=inputs, results=results)

  def to_string(self) -> str:
    """Return MLIR string.

    Returns:
        str: The function type string.
    """
    in_str = ", ".join(t.to_string() for t in self.inputs)
    if len(self.results) == 0:
      out_str = "()"
    elif len(self.results) == 1:
      out_str = self.results[0].to_string()
    else:
      out_str = "(" + ", ".join(t.to_string() for t in self.results) + ")"
    return f"({in_str}) -> {out_str}"
