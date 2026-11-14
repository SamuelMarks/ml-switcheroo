"""MLIR Type System.

This module defines strong typing structures for MLIR types, specifically focusing
on the types required for StableHLO operations (Tensors, Elements, etc.).
"""

import typing
from abc import ABC, abstractmethod

import pydantic
from ml_switcheroo_ir import SymInt, SymNode

from ml_switcheroo.core.cst.base import CstFieldValue

_DTYPE_TO_MLIR_ELEMENT: dict[str, str] = {
  "float16": "f16",
  "f16": "f16",
  "bfloat16": "bf16",
  "bf16": "bf16",
  "float32": "f32",
  "f32": "f32",
  "float64": "f64",
  "f64": "f64",
  "int8": "i8",
  "i8": "i8",
  "int16": "i16",
  "i16": "i16",
  "int32": "i32",
  "i32": "i32",
  "int64": "i64",
  "i64": "i64",
  "bool": "i1",
  "i1": "i1",
}


def format_mlir_tensor_type(
  shape: typing.Sequence[int | float | str | SymNode | SymInt | None] | None,
  dtype: str = "float32",
) -> str:
  """Format a shape tuple and dtype string into an MLIR/StableHLO ``tensor<...>`` type string.

  Args:
      shape: Sequence of integer/symbolic dimensions, or ``None`` for an unranked tensor.
      dtype: Runtime dtype string (e.g. ``'float32'``, ``'int32'``, ``'bool'``).

  Returns:
      Formatted MLIR tensor type string (e.g. ``'tensor<2x16xf32>'`` or ``'tensor<*xf32>'``).
  """
  elem_str = _DTYPE_TO_MLIR_ELEMENT.get(str(dtype).lower().strip(), "f32")
  if elem_str.startswith("i"):
    elem_type: MLIRType = IntegerType(int(elem_str[1:]))
  else:
    elem_type = FloatType(elem_str)
  shape_list: list[int | str] | None = [d if isinstance(d, int) else str(d) for d in shape] if shape is not None else None
  return TensorType(element_type=elem_type, shape=shape_list).to_string()


class MLIRType(pydantic.BaseModel, ABC):
  """Abstract base class for all strict MLIR types."""

  model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

  def __init__(self, **kwargs: CstFieldValue) -> None:
    """Initialize MLIRType.

    Args:
        **kwargs: Subclass field values forwarded to BaseModel.
    """
    super().__init__(**kwargs)

  @abstractmethod
  def to_string(self) -> str:
    """Return the MLIR string representation of the type.

    Returns:
        str: The MLIR type string.
    """


class IntegerType(MLIRType):
  """Represent an MLIR Integer Type (e.g., i1, i32, i64).

  Attributes:
      width: The width (`int`).
  """

  width: int

  def __init__(self, width: int) -> None:
    """Initialize IntegerType.

    Args:
        width: Bit width of the integer type.
    """
    super().__init__(width=width)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'i32').

    Returns:
        str: The integer type string.
    """
    return f"i{self.width}"


class FloatType(MLIRType):
  """Represent an MLIR Float Type (e.g., f32, f64, bf16).

  Attributes:
      name: The name (`str`).
  """

  name: str

  def __init__(self, name: str) -> None:
    """Initialize FloatType.

    Args:
        name: MLIR float type name.
    """
    super().__init__(name=name)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'f32').

    Returns:
        str: The float type string.
    """
    return self.name


class ComplexType(MLIRType):
  """Represent an MLIR Complex Type (e.g., complex<f32>).

  Attributes:
      element_type: The element type (`FloatType`).
  """

  element_type: FloatType

  def __init__(self, element_type: FloatType) -> None:
    """Initialize ComplexType.

    Args:
        element_type: Element FloatType of the complex type.
    """
    super().__init__(element_type=element_type)

  def to_string(self) -> str:
    """Return MLIR string (e.g. 'complex<f32>').

    Returns:
        str: The complex type string.
    """
    return f"complex<{self.element_type.to_string()}>"


class TensorType(MLIRType):
  """Represent an MLIR Tensor Type (e.g., tensor<2x?xi32>).

  Attributes:
      element_type: The element type (`MLIRType`).
      shape: The shape (`list[int | str] | None`).
  """

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
    super().__init__(element_type=element_type, shape=shape)

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
  """Represent an MLIR Function Type (e.g., (tensor<f32>) -> tensor<f32>).

  Attributes:
      inputs: The inputs (`list[MLIRType]`).
      results: The results (`list[MLIRType]`).
  """

  inputs: list[MLIRType]
  results: list[MLIRType]

  def __init__(self, inputs: list[MLIRType], results: list[MLIRType]) -> None:
    """Initialize FunctionType.

    Args:
        inputs: List of input MLIRType instances.
        results: List of result MLIRType instances.
    """
    super().__init__(inputs=inputs, results=results)

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
