"""
Inspection Engine for Python Packages.

This module provides the :class:`ApiInspector`, a hybrid static-dynamic analysis tool
designed to extract API signatures (functions, classes, and attributes) from Python packages.

It employs a two-stage strategy:

1.  **Static Analysis (Griffe)**: Attempts to parse the source code first. This is safer
    and preserves comments/docstrings better, but fails on compiled modules (C-Extensions).
2.  **Runtime Introspection (Inspect)**: Falls back to importing the module and inspecting
    live objects. This is required for frameworks like PyTorch, TensorFlow, and MLX which
    rely heavily on compiled bindings.

**Memory Safety**:
Includes recursion safeguards (visited set tracking) to prevent infinite loops
or memory explosion when traversing circular references in large libraries like PyTorch.
It also blacklists known heavy internals (`_C`, `distributed`, `cuda`) to prevent hangs during scanning.
"""

import inspect
import importlib
import logging
from typing import Dict, Any, Set

import griffe

# Suppress Griffe errors which are often noisy static analysis failures
logging.getLogger("griffe").setLevel(logging.CRITICAL)


class ApiInspector:
  """
  A robust inspector for discovering API surfaces of installed libraries.

  Attributes:
      _package_cache (Dict[str, griffe.Object]): Cache of statically parsed Griffe trees.
  """

  def __init__(self):
    """Initializes the Inspector with an empty cache."""
    self._package_cache = {}

  def inspect(self, package_name: str) -> Dict[str, Any]:
    """
    Scans a package and returns a flat catalog of its public API.

    Args:
        package_name: The importable name of the package.

    Returns:
        Dict mapping 'fully.qualified.name' -> {metadata_dict}.
    """
    catalog = {}

    # Strategy 1: Static Analysis (Best for Source Code / Docstrings)
    try:
      if package_name in self._package_cache:
        root = self._package_cache[package_name]
      else:
        root = griffe.load(package_name)
        self._package_cache[package_name] = root

      self._recurse_griffe(root, catalog)
      if len(catalog) > 0:
        return catalog
    except Exception:
      # Griffe failed (common for C-ext or complex dynamic imports).
      # We ignore it and proceed to runtime inspection.
      pass

    # Strategy 2: Runtime Inspection (Fallback for C-Extensions)
    try:
      module = importlib.import_module(package_name)
      # Track visited object IDs to prevent infinite recursion/memory explosion
      visited_ids = set()
      self._recurse_runtime(module, package_name, catalog, visited=visited_ids)
    except ImportError:
      print(f"⚠️  Could not load package '{package_name}'. Is it installed?")
    except Exception as e:
      print(f"⚠️  Error analyzing '{package_name}': {e}")

    return catalog

  def _recurse_griffe(self, obj: griffe.Object, catalog: Dict[str, Any]):
    """
    Recursively walks a Griffe object tree to build the catalog.
    """
    for member_name, member in obj.members.items():
      if member_name.startswith("_"):
        continue

      try:
        if member.is_alias:
          continue

        if member.is_function:
          info = self._extract_griffe_sig(member, kind="function")
          catalog[member.path] = info

        elif member.is_class:
          info = self._extract_griffe_sig(member, kind="class")
          catalog[member.path] = info
          self._recurse_griffe(member, catalog)

        elif member.is_attribute:
          info = self._extract_attribute_info(member)
          catalog[member.path] = info

        elif member.is_module:
          self._recurse_griffe(member, catalog)

      except Exception:
        continue

  def _recurse_runtime(self, obj: Any, path: str, catalog: Dict[str, Any], visited: Set[int], depth: int = 0):
    """
    Recursively walks a live Python object to build the catalog.

    Args:
        obj: The object to inspect.
        path: The current dot-path (e.g. 'torch.nn').
        catalog: The accumulator dict.
        visited: Set of object IDs already processed.
        depth: Recursion depth tracker.
    """
    # Safety Checks
    if depth > 5:
      return

    # Cycle detection
    obj_id = id(obj)
    if obj_id in visited:
      return
    visited.add(obj_id)

    try:
      members = inspect.getmembers(obj)
    except Exception:
      return

    # Blacklist metadata for internal modules that cause crashes during inspection
    # Updates: added '_C' (Torch binaries), 'distributed', 'cuda' (Prevent Hardware Init hangs)
    unsafe_internals = {
      "core",
      "python",
      "compiler",
      "contrib",
      "examples",
      "tools",
      "pywrap_tensorflow",
      "_logging",
      "_C",
      "distributed",
      "cuda",
      "backends",
      "fx",
      "masked",
      "ao",
      "quantization",
      "testing",
    }

    for name, member in members:
      if name.startswith("_"):
        continue

      if name in unsafe_internals:
        continue

      new_path = f"{path}.{name}"

      if inspect.ismodule(member):
        # Check integrity of package membership to avoid escaping to site-packages root
        # Also ensure we don't re-scan the same module object if encountered via alias
        if id(member) in visited:
          continue

        if hasattr(member, "__package__") and member.__package__ and path.split(".")[0] in member.__package__:
          self._recurse_runtime(member, new_path, catalog, visited, depth + 1)

      elif inspect.isclass(member) or inspect.isfunction(member) or inspect.isbuiltin(member) or inspect.ismethod(member):
        try:
          if hasattr(member, "__module__") and member.__module__:
            root_pkg = path.split(".")[0]
            # Ensure we only catalog objects belonging to this package family
            if not member.__module__.startswith(root_pkg):
              continue

          cat_type = "class" if inspect.isclass(member) else "function"
          catalog[new_path] = self._extract_runtime_sig(member, name, cat_type)

          if inspect.isclass(member):
            # Recurse into classes to find methods/nested classes
            # but check visited to allow it if first time seeing this class
            self._recurse_runtime(member, new_path, catalog, visited, depth + 1)
        except Exception:
          # C-ext fallback
          catalog[new_path] = {
            "name": name,
            "type": "function" if not inspect.isclass(member) else "class",
            "params": ["x"],
            "has_varargs": False,
            "docstring_summary": inspect.getdoc(member) or "",
          }
      # Runtime Attribute Handling (Basic check for simple types)
      elif not inspect.ismodule(member) and not inspect.isroutine(member) and not inspect.isclass(member):
        # Constants like math.pi
        if isinstance(member, (int, float, str, bool)):
          catalog[new_path] = {
            "name": name,
            "type": "attribute",
            "params": [],
            "has_varargs": False,
            "docstring_summary": f"Constant: {member}",
          }

  def _extract_griffe_sig(self, func: griffe.Object, kind: str) -> Dict[str, Any]:
    params = []
    has_varargs = False
    try:
      if hasattr(func, "parameters") and func.parameters:
        for param in func.parameters:
          p_kind = str(param.kind).lower() if param.kind else ""
          if "var_positional" in p_kind or param.name.startswith("*"):
            has_varargs = True
          clean_name = param.name.lstrip("*")
          params.append(clean_name)
    except AttributeError:
      pass

    return {
      "name": func.name,
      "type": kind,
      "params": params,
      "has_varargs": has_varargs,
      "docstring_summary": self._get_doc_summary(func),
    }

  def _extract_attribute_info(self, attr: griffe.Attribute) -> Dict[str, Any]:
    """
    Serializes an Attribute (Constant/Type).
    """
    return {
      "name": attr.name,
      "type": "attribute",
      "params": [],
      "has_varargs": False,
      "docstring_summary": self._get_doc_summary(attr),
    }

  def _extract_runtime_sig(self, obj: Any, name: str, kind: str) -> Dict[str, Any]:
    params = []
    has_varargs = False
    try:
      sig = inspect.signature(obj)
      for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
          has_varargs = True
        params.append(p.name)
    except (ValueError, TypeError):
      pass

    return {
      "name": name,
      "type": kind,
      "params": params,
      "has_varargs": has_varargs,
      "docstring_summary": inspect.getdoc(obj) or "",
    }

  def _get_doc_summary(self, obj: Any) -> str:
    doc = ""
    if hasattr(obj, "docstring") and obj.docstring:
      full_doc = obj.docstring.value or ""
      doc = full_doc.strip().split("\n")[0]
    elif hasattr(obj, "__doc__") and obj.__doc__:
      doc = str(obj.__doc__).strip().split("\n")[0]
    return doc
