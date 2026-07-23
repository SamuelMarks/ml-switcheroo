"""MLIR Type System.

This module defines strong typing structures for MLIR types, specifically focusing
on the types required for StableHLO operations (Tensors, Elements, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union


class MLIRType(ABC):
  """Abstract base class for all strict MLIR types."""

  @abstractmethod
  def to_string(self) -> str:
    """Return the MLIR string representation of the type."""


@dataclass(frozen=True)
class IntegerType(MLIRType):
  """Represents an MLIR Integer Type (e.g., i1, i32, i64)."""

  width: int

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'i32')."""
    return f"i{self.width}"


@dataclass(frozen=True)
class FloatType(MLIRType):
  """Represents an MLIR Float Type (e.g., f32, f64, bf16)."""

  name: str

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'f32')."""
    return self.name


@dataclass(frozen=True)
class ComplexType(MLIRType):
  """Represents an MLIR Complex Type (e.g., complex<f32>)."""

  element_type: FloatType

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'complex<f32>')."""
    return f"complex<{self.element_type.to_string()}>"


@dataclass(frozen=True)
class TensorType(MLIRType):
  """Represents an MLIR Tensor Type (e.g., tensor<2x?xi32>)."""

  element_type: MLIRType
  shape: Optional[List[Union[int, str]]] = None  # None means unranked (*), '?' means dynamic dimension

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'tensor<2x?xf32>')."""
    if self.shape is None:
      return f"tensor<*x{self.element_type.to_string()}>"

    if len(self.shape) == 0:
      return f"tensor<{self.element_type.to_string()}>"

    shape_strs = [str(dim) for dim in self.shape]
    shape_prefix = "x".join(shape_strs)
    return f"tensor<{shape_prefix}x{self.element_type.to_string()}>"


@dataclass(frozen=True)
class FunctionType(MLIRType):
  """Represents an MLIR Function Type (e.g., (tensor<f32>) -> tensor<f32>)."""

  inputs: List[MLIRType]
  results: List[MLIRType]

  def to_string(self) -> str:
    """Return MLIR string."""
    in_str = ", ".join(t.to_string() for t in self.inputs)
    if len(self.results) == 0:
      out_str = "()"
    elif len(self.results) == 1:
      out_str = self.results[0].to_string()
    else:
      out_str = "(" + ", ".join(t.to_string() for t in self.results) + ")"
    return f"({in_str}) -> {out_str}"
