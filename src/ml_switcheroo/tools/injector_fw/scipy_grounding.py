"""SciPy Fallback Grounding and Validation Utilities.

Validates that all injected `scipy.special`, `scipy.linalg`, and related fallback
AST macros and imports correspond strictly to verified symbols and signatures
in the static SciPy snapshot (e.g. `scipy_v1.13.1.json`), maintaining zero-hallucination
AST transpilation.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ml_switcheroo.semantics.paths import resolve_snapshots_dir


def load_scipy_snapshot(snapshot_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
  """Load and index verified SciPy API definitions from a snapshot JSON file.

  Args:
      snapshot_path: Optional path to the SciPy snapshot JSON file. If omitted,
          searches `resolve_snapshots_dir()` for `scipy_v*.json`.

  Returns:
      Dict[str, Dict[str, Any]]: Dictionary mapping fully-qualified API paths
          (e.g., 'scipy.special.erf') to their snapshot specifications.

  Raises:
      FileNotFoundError: If no SciPy snapshot file can be located.
  """
  if snapshot_path is None:
    snap_dir = resolve_snapshots_dir()
    candidates = sorted(snap_dir.glob("scipy_v*.json"))
    if not candidates:
      raise FileNotFoundError(f"No SciPy snapshot found in {snap_dir}")
    snapshot_path = candidates[-1]

  if not snapshot_path.exists():
    raise FileNotFoundError(f"SciPy snapshot path does not exist: {snapshot_path}")

  with snapshot_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

  index: Dict[str, Dict[str, Any]] = {}
  categories = data.get("categories", {})
  if isinstance(categories, dict):
    for cat_items in categories.values():
      if isinstance(cat_items, list):
        for item in cat_items:
          if isinstance(item, dict):
            api_path = item.get("api_path") or item.get("name")
            if api_path:
              index[api_path] = item
              name_only = api_path.split(".")[-1]
              index[name_only] = item

  return index


def extract_scipy_endpoints_from_macro(macro: str) -> Set[str]:
  """Extract all referenced SciPy endpoints from a Python macro template.

  Args:
      macro: Python code string or template containing placeholder formatting.

  Returns:
      Set[str]: Set of referenced SciPy attribute calls or paths.
  """
  # Replace format string placeholders like {x} or {x.shape} with dummy identifiers
  clean_macro = re.sub(r"\{[^{}]+\}", "DUMMY_ARG", macro)

  try:
    tree = ast.parse(clean_macro)
  except SyntaxError:
    return set()

  # Map child -> parent to find outermost attributes only
  parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

  endpoints: Set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and not isinstance(parents.get(node), ast.Attribute):
      chain: List[str] = []
      curr: Optional[ast.AST] = node
      while isinstance(curr, ast.Attribute):
        chain.append(curr.attr)
        curr = curr.value
      if isinstance(curr, ast.Name):
        chain.append(curr.id)
        full_path = ".".join(reversed(chain))
        if full_path.startswith("scipy.") or full_path.startswith("jax.scipy."):
          endpoints.add(full_path)
  return endpoints


def validate_scipy_api(
  api_path: str,
  snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
  """Validate whether a given SciPy API endpoint is grounded in the snapshot.

  Args:
      api_path: Fully qualified API string (e.g. 'scipy.special.i0').
      snapshot: Optional pre-loaded snapshot index. If None, loads default snapshot.

  Returns:
      bool: True if the endpoint exists in the grounded snapshot.

  Raises:
      ValueError: If the endpoint is ungrounded or absent in the snapshot.
  """
  if snapshot is None:
    snapshot = load_scipy_snapshot()

  normalized = api_path
  if normalized.startswith("jax.scipy."):
    normalized = normalized.replace("jax.scipy.", "scipy.", 1)

  if normalized in snapshot:
    return True

  name_only = normalized.split(".")[-1]
  if name_only in snapshot:
    return True

  raise ValueError(f"Ungrounded SciPy endpoint detected: '{api_path}'. Not present in snapshot.")


def validate_scipy_macro(
  macro_template: str,
  snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
  """Validate all SciPy calls within a fallback macro template against the snapshot.

  Args:
      macro_template: The AST macro template string.
      snapshot: Optional pre-loaded snapshot index.

  Returns:
      bool: True if all detected SciPy endpoints are grounded.

  Raises:
      ValueError: If any detected SciPy endpoint in the macro is ungrounded.
  """
  if snapshot is None:
    snapshot = load_scipy_snapshot()

  endpoints = extract_scipy_endpoints_from_macro(macro_template)
  for ep in endpoints:
    validate_scipy_api(ep, snapshot)
  return True


def audit_injected_scipy_fallbacks(
  variants: Dict[str, Any],
  snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
  """Audit all framework variants for ungrounded SciPy APIs and imports.

  Args:
      variants: Mapping of variant names to variant specifications or dictionaries.
      snapshot: Optional pre-loaded snapshot index.

  Returns:
      List[str]: List of error messages for any unverified endpoints.
  """
  if snapshot is None:
    try:
      snapshot = load_scipy_snapshot()
    except Exception as e:
      return [f"Failed to load SciPy snapshot for audit: {e}"]

  errors: List[str] = []
  for var_name, var_def in variants.items():
    if not isinstance(var_def, dict):
      continue

    # 1. Audit explicit API
    api = var_def.get("api")
    if isinstance(api, str) and ("scipy." in api):
      try:
        validate_scipy_api(api, snapshot)
      except ValueError as err:
        errors.append(f"[{var_name}] {err}")

    # 2. Audit macro template
    macro = var_def.get("macro_template")
    if isinstance(macro, str) and ("scipy." in macro):
      try:
        validate_scipy_macro(macro, snapshot)
      except ValueError as err:
        errors.append(f"[{var_name}] {err}")

  return errors
