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
from typing import Any, Callable, Dict, List, Optional, Union

from ml_switcheroo_ir.schema.ghost import GhostParam as _BaseGhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef as _BaseGhostRef
from pydantic import ConfigDict, Field, model_validator


class GhostParam(_BaseGhostParam):
  """Serializable representation of a function parameter."""

  model_config = ConfigDict(extra="allow", populate_by_name=True)

  kind: Any = Field(default="POSITIONAL_OR_KEYWORD", description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD).")


class GhostRef(_BaseGhostRef):
  """Serializable representation of a Framework API component.

  Replaces live Python objects in 'Ghost Mode'.
  """

  model_config = ConfigDict(extra="allow", populate_by_name=True)

  api_path: str = Field(default="", description="Fully qualified import path.")
  kind: str = Field(default="function", description="One of: 'class', 'function'")
  params: List[Any] = Field(default_factory=list, description="List of parameters.")
  api: Optional[str] = Field(None, exclude=True, description="Alias for api_path.")
  group: Optional[str] = Field(None, exclude=True, description="Optional category group.")

  @model_validator(mode="before")
  @classmethod
  def _normalize_legacy_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy `api` keyword argument to `api_path` and hydrate parameter dicts.

    Args:
        data: Input dictionary.

    Returns:
        Normalized input dictionary.
    """
    data = dict(data)
    if data.get("api") and not data.get("api_path"):
      data["api_path"] = data["api"]
    if "params" in data:
      data["params"] = [GhostParam.model_validate(p) if isinstance(p, dict) else p for p in data["params"]]
    return data


class GhostInspector:
  """Facade for API Inspection.

  Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
  Crucial for populating snapshots used by WASM/JS environments.
  """

  @staticmethod
  def inspect(obj: Union[type, Callable[..., Any]], api_path: str) -> "GhostRef":
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
    params = []
    has_varargs = False

    # Determine target for signature extraction
    # If class, we want __init__. If function, we want obj itself.
    target = obj
    if kind == "class":
      target = getattr(obj, "__init__", obj)

    try:
      # 1. Standard Introspection
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
            except Exception:
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
            kind=str(param.kind),
            default=default_val,
            annotation=anno_val,
          )
        )

    except (ValueError, TypeError):
      # 2. C-Extension Fallback
      # Many ML library functions are compiled C++ and lack python signatures.
      # We check the docstring or assume generic arguments.

      # Simple heuristic: If it's a function and we can't inspect it,
      # assume it takes args/kwargs to allow it to pass consensus checks.
      # This is better than returning empty params which implies 0-arity.
      if kind == "function":
        has_varargs = True
        params.append(GhostParam(name="args", kind="VAR_POSITIONAL"))
        params.append(GhostParam(name="kwargs", kind="VAR_KEYWORD"))

    return GhostRef(name=name, api_path=api_path, kind=kind, params=params, docstring=doc, has_varargs=has_varargs)

  @staticmethod
  def hydrate(data: dict[Any, Any]) -> "GhostRef":
    """Create a GhostRef from a dictionary (JSON snapshot).

    Args:
        data: The dictionary data.

    Returns:
        The hydrated GhostRef object.

    """
    return GhostRef.model_validate(data)
