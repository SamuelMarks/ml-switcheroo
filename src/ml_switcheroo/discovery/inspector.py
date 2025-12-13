"""
Inspection Engine for Python Packages.

This module uses ``griffe`` to perform static analysis on installed packages,
extracting function signatures, docstrings, and identifying attributes (constants/types).
It provides robust detection of variable arguments (``*args``) to aid fuzzy matching.
"""

from typing import Dict, Any
import griffe


class ApiInspector:
  """
  Static Analysis tool for Python packages.
  """

  def __init__(self):
    """Initializes the Inspector with an empty cache."""
    self._package_cache = {}

  def inspect(self, package_name: str) -> Dict[str, Any]:
    """
    Scans a package and returns a flat catalog of its public API.

    Args:
        package_name: The importable name of the package (e.g. 'torch', 'math').

    Returns:
        Dict[str, Any]: Catalog of API members.
        Format: ``{ "pkg.sub.func": { "name": "func", "has_varargs": True, ... } }``
    """
    try:
      # Griffe loads the package tree
      if package_name in self._package_cache:
        root_module = self._package_cache[package_name]
      else:
        root_module = griffe.load(package_name)
        self._package_cache[package_name] = root_module
    except ImportError:
      print(f"⚠️  Could not load package '{package_name}'. Is it installed?")
      return {}
    except Exception as e:
      print(f"⚠️  Error analyzing package '{package_name}': {e}")
      return {}

    catalog = {}
    self._recurse(root_module, catalog)
    return catalog

  def _recurse(self, obj: griffe.Object, catalog: Dict[str, Any]):
    """
    Recursively walks modules and classes.

    Args:
        obj: The current Griffe object being visited.
        catalog: The accumulator dictionary for API definitions.
    """
    for member_name, member in obj.members.items():
      # Skip private members (heuristic)
      if member_name.startswith("_") and not member_name.startswith("__"):
        continue

      try:
        # 1. Alias Handling
        if member.is_alias:
          # We typically skip aliases to avoid duplication,
          # but if it's a re-export of a constant, it might be relevant.
          # For stability, we skip unless we verify target is external.
          continue

        # 2. Functions
        if member.is_function:
          info = self._extract_signature(member, kind="function")
          catalog[member.path] = info

        # 3. Attributes (Constants/Types) -> Feature 015
        elif member.is_attribute:
          info = self._extract_attribute_info(member)
          catalog[member.path] = info

        # 4. Classes (Recurse + Catalog Class Itself)
        elif member.is_class:
          # Catalog the class itself (so we map 'Linear' not just 'Linear.forward')
          info = self._extract_signature(member, kind="class")
          catalog[member.path] = info

          # Recurse into methods/nested classes
          self._recurse(member, catalog)

        # 5. Modules (Recurse only)
        elif member.is_module:
          self._recurse(member, catalog)

      except Exception:
        # If inspection fails for a specific member, skip gracefully
        continue

  def _extract_signature(self, func: griffe.Object, kind: str) -> Dict[str, Any]:
    """
    Serializes a Function or Class signature.

    Detects standard parameters and varargs (``*args``).

    Args:
        func: The Griffe function/class object.
        kind: Label string ('function' or 'class').

    Returns:
        Dictionary containing metadata: name, type, params, has_varargs, docstring.
    """
    params = []
    has_varargs = False

    # Access parameters safely
    try:
      if hasattr(func, "parameters") and func.parameters:
        for param in func.parameters:
          # Feature 07: Robust support for *args detection.
          # Griffe 0.30+ uses .kind enum. We converting to string to be safe.
          # "var_positional" corresponds to *args.
          p_kind = str(param.kind).lower() if param.kind else ""

          if "var_positional" in p_kind or (param.name.startswith("*") and not param.name.startswith("**")):
            has_varargs = True

          # We store the cleaned name for the parameter list
          clean_name = param.name.lstrip("*")
          params.append(clean_name)
    except AttributeError:
      pass

    return {
      "name": func.name,
      "type": kind,  # 'function' or 'class'
      "params": params,
      "has_varargs": has_varargs,
      "docstring_summary": self._get_doc_summary(func),
    }

  def _extract_attribute_info(self, attr: griffe.Attribute) -> Dict[str, Any]:
    """
    Serializes an Attribute (Constant/Type).

    Args:
        attr: Griffe Attribute object.

    Returns:
        Dictionary identifying the object as an attribute.
    """
    return {
      "name": attr.name,
      "type": "attribute",
      "params": [],  # Attributes have no parameters
      "has_varargs": False,
      "docstring_summary": self._get_doc_summary(attr),
    }

  def _get_doc_summary(self, obj: griffe.Object) -> str:
    """
    Helper to safely extract the first line of docstring.

    Args:
        obj: The griffe object.

    Returns:
        The first non-empty line of the docstring or empty string.
    """
    doc = ""
    if obj.docstring:
      # Griffe docstrings are objects, .value gets the raw text
      full_doc = obj.docstring.value or ""
      doc = full_doc.strip().split("\n")[0]
    return doc
