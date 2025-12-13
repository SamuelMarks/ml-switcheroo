"""
Ghost Core: Introspection Abstraction Layer.

This module provides the data structures and inspection logic required to
decouple framework analysis from the live environment. It enables the system
to operate in "Ghost Mode" (WASM/CI) by working against cached snapshots
instead of requiring heavy libraries (Torch/TensorFlow) to be installed.

Classes:
    GhostParam: Represents a single function/method parameter.
    GhostRef: A serializable snapshot of an API component (Class or Function).
    GhostInspector: Facade to extract GhostRefs from live objects or JSON.
"""

import inspect
from typing import Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field


class GhostParam(BaseModel):
  """
  Serializable representation of a function parameter.
  """

  name: str
  kind: str
  default: Optional[str] = None
  annotation: Optional[str] = None


class GhostRef(BaseModel):
  """
  Serializable snapshot of a Framework API component.

  Used by the Consensus Engine to align concepts (e.g. 'HuberLoss' vs 'Huber')
  without needing the framework installed at runtime.
  """

  name: str
  api_path: str
  kind: str = Field(description="One of: 'class', 'function'")
  params: List[GhostParam] = Field(default_factory=list)
  docstring: Optional[str] = None

  def has_arg(self, arg_name: str) -> bool:
    """Checks if a specific argument exists in the signature."""
    return any(p.name == arg_name for p in self.params)


class GhostInspector:
  """
  Facade for API Inspection.

  Handles the complexity of:
  1.  Live Introspection: Using ``inspect.signature`` on installed libraries.
  2.  Ghost Hydration: wrapper for Pydantic loading (used for future caching interactions).
  3.  Normalization: Converting classes (inspecting ``__init__``) transparently.
  """

  @staticmethod
  def inspect(obj: Union[Any, Callable], api_path: str) -> "GhostRef":
    """
    Creates a GhostRef from a live Python object.

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

    # Determine target for signature extraction
    # If class, we want __init__. If function, we want obj itself.
    target = obj
    if kind == "class":
      target = getattr(obj, "__init__", obj)

    try:
      sig = inspect.signature(target)
      for param in sig.parameters.values():
        # Skip 'self' for methods
        if param.name == "self":
          continue

        # Serialize defaults safely
        default_val = None
        if param.default is not inspect.Parameter.empty:
          default_val = str(param.default)

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
      # Built-ins (C-extensions) might fail signature inspection
      pass

    return GhostRef(
      name=name,
      api_path=api_path,
      kind=kind,
      params=params,
      docstring=doc,
    )

  @staticmethod
  def hydrate(data: dict) -> "GhostRef":
    """
    Creates a GhostRef from a dictionary (JSON snapshot).

    Args:
        data: Dictionary matching the GhostRef schema.

    Returns:
        A valid GhostRef object.
    """
    return GhostRef.model_validate(data)
