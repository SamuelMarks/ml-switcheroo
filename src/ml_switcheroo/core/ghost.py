"""
Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and inspection logic required to
decouple framework analysis from the live environment. It enables the system
to operate in "Ghost Mode" (WASM/CI) by working against cached snapshots
instead of requiring heavy libraries (Torch/TensorFlow) to be installed.

Updates:
- Robust C-Extension handling (try/except around `inspect.signature`).
- Validates parameter kinds to support `*args` (VarPositional).
"""

import inspect
from typing import Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field


class GhostParam(BaseModel):
  """
  Serializable representation of a function parameter.
  """

  name: str = Field(description="Parameter name.")
  kind: str = Field(description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD).")
  default: Optional[str] = Field(None, description="Default value as string.")
  annotation: Optional[str] = Field(None, description="Type annotation as string.")


class GhostRef(BaseModel):
  """
  Serializable snapshot of a Framework API component.

  Used by the Consensus Engine to align concepts (e.g. 'HuberLoss' vs 'Huber')
  without needing the framework installed at runtime.
  """

  name: str = Field(description="Short name of the object.")
  api_path: str = Field(description="Fully qualified import path.")
  kind: str = Field(description="One of: 'class', 'function'")
  params: List[GhostParam] = Field(default_factory=list, description="List of parameters.")
  docstring: Optional[str] = Field(None, description="Extracted docstring.")
  has_varargs: bool = Field(False, description="True if signature accepts *args.")

  def has_arg(self, arg_name: str) -> bool:
    """
    Checks if a specific argument exists in the signature.

    Args:
        arg_name: The argument name to find.

    Returns:
        True if found.
    """
    return any(p.name == arg_name for p in self.params)


class GhostInspector:
  """
  Facade for API Inspection.

  Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
  Crucial for populating snapshots used by WASM/JS environments.
  """

  @staticmethod
  def inspect(obj: Union[Any, Callable], api_path: str) -> "GhostRef":
    """
    Creates a GhostRef from a live Python object.
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
          try:
            default_val = str(param.default)
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
  def hydrate(data: dict) -> "GhostRef":
    """
    Creates a GhostRef from a dictionary (JSON snapshot).

    Args:
        data: The dictionary data.

    Returns:
        The hydrated GhostRef object.
    """
    return GhostRef.model_validate(data)
