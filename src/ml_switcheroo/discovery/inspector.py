"""Introspection Type Definitions and Helpers for API Discovery.

Provides the canonical `InspectionObject` type alias and helper predicates
for inspecting live Python modules, classes, functions, builtins, descriptors,
and properties during API discovery.
"""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo.enums import InspectionObject as InspectionObject


def is_inspection_object(obj: object) -> typing.TypeGuard[InspectionObject]:
  """Check whether a Python object is a supported live inspection target.

  Args:
      obj: Candidate Python object to check.

  Returns:
      True if `obj` is a module, class, function, method, builtin, method descriptor,
      or property.
  """
  return isinstance(
    obj,
    (
      types.ModuleType,
      type,
      types.FunctionType,
      types.MethodType,
      types.BuiltinFunctionType,
      types.MethodDescriptorType,
      property,
    ),
  ) or inspect.isroutine(obj)
