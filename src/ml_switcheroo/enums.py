"""Enumeration and Core Domain Type Aliases for ml-switcheroo.

This module defines standard enumerations and structural protocols used across
the codebase for semantic categorization, framework identification, runtime
tensor/value conversion, and live object introspection.
"""

from __future__ import annotations

import types
import typing
from enum import Enum

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


@typing.runtime_checkable
class _CustomInstanceProtocol(typing.Protocol):
  """Protocol matching user-defined Python class instances (which carry `__module__`)."""

  __module__: str


@typing.runtime_checkable
class CallableTargetProtocol(typing.Protocol):
  """Protocol for arbitrary callable runtime targets."""

  def __call__(self, *args: "RuntimeTensorOrValue", **kwargs: "RuntimeTensorOrValue") -> "RuntimeTensorOrValue":
    """Invoke callable target.

    Args:
        *args: Positional runtime arguments.
        **kwargs: Keyword runtime arguments.

    Returns:
        Resulting runtime tensor or value.
    """


_CallableProtocol = CallableTargetProtocol

InspectionObject: typing.TypeAlias = (
  types.ModuleType
  | _ClassProtocol
  | _CustomInstanceProtocol
  | _CallableProtocol
  | types.FunctionType
  | types.BuiltinFunctionType
  | types.MethodType
  | property
  | Enum
  | BaseModel
  | str
  | int
  | float
  | bool
  | None
)


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
  | types.EllipsisType
  | BaseModel
  | Enum
  | BaseException
  | types.ModuleType
  | _ClassProtocol
  | _CustomInstanceProtocol
  | _CallableProtocol
  | types.FunctionType
  | types.BuiltinFunctionType
  | types.MethodType
  | typing.Callable[[JsonValue], JsonValue]
  | typing.Sequence["RuntimeTensorOrValue"]
  | typing.Mapping[str, "RuntimeTensorOrValue"]
  | typing.Collection["RuntimeTensorOrValue"]
  | None
)

FuzzValue: typing.TypeAlias = RuntimeTensorOrValue
