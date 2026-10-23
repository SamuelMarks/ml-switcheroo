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

  def __array__(self) -> object:
    """Return an array representation of the object.

    Returns:
        Array-compatible representation.
    """


@typing.runtime_checkable
class DetachableProtocol(typing.Protocol):
  """Protocol for framework tensor objects exposing ``.detach()``."""

  def detach(self) -> object:
    """Return a detached tensor view without gradient tracking.

    Returns:
        Detached tensor object.
    """


@typing.runtime_checkable
class NumpyConvertibleProtocol(typing.Protocol):
  """Protocol for framework tensor objects exposing ``.numpy()``."""

  def numpy(self) -> object:
    """Convert the tensor into a NumPy array or Python collection.

    Returns:
        Converted NumPy array or Python value.
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

InspectionObject: typing.TypeAlias = (
  types.ModuleType
  | type
  | types.FunctionType
  | types.MethodType
  | types.BuiltinFunctionType
  | types.MethodDescriptorType
  | property
)

RuntimeTensorOrValue: typing.TypeAlias = (
  ArrayLikeProtocol
  | JsonValue
  | int
  | float
  | complex
  | bool
  | str
  | None
  | types.EllipsisType
  | BaseModel
  | InspectionObject
  | typing.Callable[[JsonValue], JsonValue]
  | typing.Sequence["RuntimeTensorOrValue"]
  | typing.Mapping[str, "RuntimeTensorOrValue"]
  | typing.Collection["RuntimeTensorOrValue"]
  | object
)

FuzzValue: typing.TypeAlias = RuntimeTensorOrValue
