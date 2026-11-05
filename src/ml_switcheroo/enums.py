"""Enumeration and Core Domain Type Aliases for ml-switcheroo.

This module defines standard enumerations and structural protocols used across
the codebase for semantic categorization, framework identification, runtime
tensor/value conversion, and live object introspection.
"""

from __future__ import annotations

from enum import Enum
import types
import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier as SemanticTier
from pydantic import BaseModel, JsonValue


class LogicOp(str, Enum):
  """Supported operators for conditional logic rules in operations.

  Used for Conditional API Dispatch.
  """

  EQ = "eq"  # ==
  NEQ = "neq"  # !=
  GT = "gt"  # >
  LT = "lt"  # <
  GTE = "gte"  # >=
  LTE = "lte"  # <=
  IN = "in"  # value in [list]
  NOT_IN = "not_in"  # value not in [list]
  IS_TYPE = "is_type"  # Checks if AST node matches type (int, list, etc)


@typing.runtime_checkable
class ArrayInterfaceProtocol(typing.Protocol):
  """Protocol for array-like objects exposing the NumPy ``__array__`` method."""

  def __array__(self) -> "RuntimeTensorOrValue":
    """Return an array representation of the object.

    Returns:
        Array-compatible representation.
    """


@typing.runtime_checkable
class NumpyConvertibleProtocol(typing.Protocol):
  """Protocol for framework tensor objects exposing ``.numpy()``."""

  def numpy(self) -> "RuntimeTensorOrValue":
    """Convert the tensor into a NumPy array or Python collection.

    Returns:
        Converted NumPy array or Python value.
    """


@typing.runtime_checkable
class CpuConvertibleProtocol(typing.Protocol):
  """Protocol for framework tensor objects exposing ``.cpu()``."""

  def cpu(self) -> NumpyConvertibleProtocol:
    """Move the tensor to CPU and return a NumPy-convertible tensor.

    Returns:
        CPU tensor exposing ``.numpy()``.
    """


@typing.runtime_checkable
class DetachableProtocol(typing.Protocol):
  """Protocol for framework tensor objects exposing ``.detach()``."""

  def detach(self) -> CpuConvertibleProtocol:
    """Return a detached tensor view without gradient tracking.

    Returns:
        Detached tensor object.
    """


@typing.runtime_checkable
class ShapeProtocol(typing.Protocol):
  """Protocol for tensor or array objects exposing a ``.shape`` attribute."""

  @property
  def shape(self) -> typing.Sequence[int | str | None]:
    """Return the tensor dimension tuple or sequence.

    Returns:
        Sequence of dimension sizes.
    """


ArrayLikeProtocol: typing.TypeAlias = (
  ArrayInterfaceProtocol | DetachableProtocol | NumpyConvertibleProtocol | ShapeProtocol
)


@typing.runtime_checkable
class _ClassProtocol(typing.Protocol):
  """Protocol for Python class objects."""

  __name__: str


InspectionObject: typing.TypeAlias = types.ModuleType | _ClassProtocol | typing.Hashable


@typing.runtime_checkable
class _SliceProtocol(typing.Protocol):
  """Protocol for Python slice objects."""

  @property
  def start(self) -> int | None:
    """Return slice start index.

    Returns:
        Start index or None.
    """

  @property
  def stop(self) -> int | None:
    """Return slice stop index.

    Returns:
        Stop index or None.
    """

  @property
  def step(self) -> int | None:
    """Return slice step.

    Returns:
        Step or None.
    """


RuntimeTensorOrValue: typing.TypeAlias = (
  ArrayLikeProtocol
  | JsonValue
  | int
  | float
  | complex
  | bool
  | str
  | _SliceProtocol
  | range
  | None
  | types.EllipsisType
  | BaseModel
  | types.ModuleType
  | _ClassProtocol
  | typing.Hashable
  | typing.Callable[[JsonValue], JsonValue]
  | typing.Sequence["RuntimeTensorOrValue"]
  | typing.Mapping[str, "RuntimeTensorOrValue"]
  | typing.Collection["RuntimeTensorOrValue"]
)

FuzzValue: typing.TypeAlias = RuntimeTensorOrValue
