"""Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and inspection logic required to
decouple framework analysis from the live environment. It enables the system
to operate in "Ghost Mode" (WASM/CI) by working against cached snapshots
instead of requiring heavy libraries (Torch/TensorFlow) to be installed.

Updates:
- Robust C-Extension handling (try/except around `inspect.signature`).
- Validates parameter kinds to support `*args` (VarPositional).
- Sanitizes default values to avoid serializing memory addresses.
"""

import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostParam as _BaseGhostParam
from ml_switcheroo_ir.schema.ghost import ParameterKind
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

InspectableTarget = typing.Union[
  type,
  types.FunctionType,
  types.MethodType,
  types.BuiltinFunctionType,
  types.ModuleType,
]


class GhostParam(_BaseGhostParam):
  """Serializable representation of a function parameter."""

  model_config = ConfigDict(extra="allow", populate_by_name=True)

  kind: ParameterKind = Field(
    default=ParameterKind.POSITIONAL_OR_KEYWORD,
    description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD).",
  )


class GhostRef(BaseModel):
  """Serializable representation of a Framework API component.

  Replaces live Python objects in 'Ghost Mode'.
  """

  model_config = ConfigDict(extra="allow", populate_by_name=True)

  name: str = Field(description="Short name (e.g. 'abs', 'Linear').")
  api_path: str = Field(default="", description="Fully qualified import path.")
  kind: str = Field(default="function", description="One of: 'class', 'function'")
  params: typing.List[GhostParam] = Field(default_factory=list, description="List of parameters.")
  docstring: typing.Optional[str] = Field(default=None, description="Extracted documentation.")
  has_varargs: bool = Field(default=False, description="True if *args is present.")
  return_annotation: typing.Optional[str] = Field(default=None, description="Return type hint.")
  api: typing.Optional[str] = Field(None, exclude=True, description="Alias for api_path.")
  group: typing.Optional[str] = Field(None, exclude=True, description="Optional category group.")

  @model_validator(mode="before")
  @classmethod
  def _normalize_legacy_fields(
    cls,
    data: typing.Union[
      typing.Dict[str, typing.Union[JsonValue, GhostParam, typing.List[GhostParam]]], typing.Mapping[str, JsonValue]
    ],
  ) -> typing.Dict[str, typing.Union[JsonValue, GhostParam, typing.List[GhostParam]]]:
    """Normalize legacy `api` keyword argument to `api_path` and hydrate parameter dicts.

    Args:
        data: Input dictionary.

    Returns:
        Normalized input dictionary.
    """
    out: typing.Dict[str, typing.Union[JsonValue, GhostParam, typing.List[GhostParam]]] = dict(data)
    if out.get("api") and not out.get("api_path"):
      out["api_path"] = out["api"]
    raw_params = out.get("params")
    if isinstance(raw_params, list):
      out["params"] = [p if isinstance(p, GhostParam) else GhostParam.model_validate(p) for p in raw_params]
    return out

  def has_arg(self, arg_name: str) -> bool:
    """Check if the signature contains a specific argument name.

    Args:
        arg_name: Name of the argument to check.

    Returns:
        True if the argument is present in `params`, False otherwise.
    """
    return any(p.name == arg_name for p in self.params)


class GhostInspector:
  """Facade for API Inspection.

  Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
  Crucial for populating snapshots used by WASM/JS environments.
  """

  @staticmethod
  def inspect(obj: typing.Union[InspectableTarget, JsonValue], api_path: str) -> "GhostRef":
    """Create a GhostRef from a live Python object.

    Gracefully handles C-Extensions and builtins that resist introspection.

    Args:
        obj: The live class or function to inspect.
        api_path: The canonical string path (e.g. 'torch.nn.ReLU').

    Returns:
        A populated GhostRef object.

    """
    name = getattr(obj, "__name__", api_path.split(".")[-1])
    kind = "class" if inspect.isclass(obj) else "function"
    doc = inspect.getdoc(obj)
    params: typing.List[GhostParam] = []
    has_varargs = False

    # Determine target for signature extraction
    # If class, we want __init__. If function, we want obj itself.
    target = obj
    if kind == "class":
      target = getattr(obj, "__init__", obj)

    try:
      # 1. Standard Introspection
      assert callable(target)
      sig = inspect.signature(target)

      for param in sig.parameters.values():
        # Skip 'self' for methods
        if param.name == "self":
          continue

        # Detect *args
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
          has_varargs = True

        # Serialize defaults safely (convert objects to string representation)
        default_val = None
        if param.default is not inspect.Parameter.empty:
          val = param.default

          # Sanitize callables or objects with memory addresses to ensure deterministic JSON
          # e.g. <function mean at 0x...> -> null
          if callable(val) or " at 0x" in repr(val):
            default_val = None
          else:
            try:
              default_val = str(val)
              # Double check string conversion didn't leak address
              if " at 0x" in default_val:
                default_val = None
            except (ValueError, TypeError, RuntimeError):
              default_val = "<unrepresentable>"

        # Serialize annotation
        anno_val = None
        if param.annotation is not inspect.Parameter.empty:
          # Handle typing objects vs string forward refs
          if hasattr(param.annotation, "__name__"):
            anno_val = param.annotation.__name__
          else:
            anno_val = str(param.annotation)

        params.append(
          GhostParam(
            name=param.name,
            kind=ParameterKind(str(param.kind)),
            default=default_val,
            annotation=anno_val,
          )
        )

    except (ValueError, TypeError, AssertionError):
      # 2. C-Extension Fallback
      # Many ML library functions are compiled C++ and lack python signatures.
      # We check the docstring or assume generic arguments.

      # Simple heuristic: If it's a function and we can't inspect it,
      # assume it takes args/kwargs to allow it to pass consensus checks.
      # This is better than returning empty params which implies 0-arity.
      if kind == "function":
        has_varargs = True
        params.append(GhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL))
        params.append(GhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD))

    return GhostRef(name=name, api_path=api_path, kind=kind, params=params, docstring=doc, has_varargs=has_varargs)

  @staticmethod
  def hydrate(data: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]) -> "GhostRef":
    """Create a GhostRef from a dictionary (JSON snapshot).

    Args:
        data: The dictionary data.

    Returns:
        The hydrated GhostRef object.

    """
    return GhostRef.model_validate(data)
