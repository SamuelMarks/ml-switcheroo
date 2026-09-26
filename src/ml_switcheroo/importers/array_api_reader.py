"""Importer for Python Array API Standard Stubs.

This module parses .py stub files from the official Array API standard repo,
extracting function signatures, type hints, and docstrings to build the
Semantic Knowledge Base.

Feature 027 Update:
Now extracts type hints (e.g. ``x: Array``, ``axis: int``) to support Better Fuzzing.
"""

from typing import Any
import typing
import ast
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from ml_switcheroo.utils.console import log_info, log_warning
from ml_switcheroo.semantics.paths import resolve_snapshots_dir


class ArrayApiSpecImporter:
  """Parse Python stub files (``*.py``) using the built-in ``ast`` module."""

  def parse_snapshot(self, snapshot_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Parse Array API specifications from a serialized snapshot JSON.

    Args:
        snapshot_path: Optional path to the snapshot JSON file. If omitted,
            resolves ``array_api_v2024.12.json`` via ``resolve_snapshots_dir()``.

    Returns:
        Dict mapping function names to semantic definitions.

    Raises:
        FileNotFoundError: If the snapshot file does not exist.
        ValueError: If the snapshot JSON is corrupt or not a dictionary.
    """
    if snapshot_path is None:
      snapshot_path = resolve_snapshots_dir() / "array_api_v2024.12.json"

    if not snapshot_path.exists():
      raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    try:
      raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as e:
      raise ValueError(f"Corrupt snapshot JSON at {snapshot_path}: {e}") from e

    if not isinstance(raw, dict):
      raise ValueError("Corrupt snapshot: top-level JSON must be an object.")

    semantics: Dict[str, Dict[str, Any]] = {}
    categories = raw.get("categories", {})
    all_items: List[Dict[str, Any]] = []

    if isinstance(categories, dict):
      for cat_items in categories.values():
        if isinstance(cat_items, list):
          all_items.extend(cat_items)

    ops = raw.get("operations", {})
    if isinstance(ops, dict):
      all_items.extend(ops.values())

    for entry in all_items:
      if not isinstance(entry, dict):
        continue
      api_path = entry.get("api_path", "")
      name = entry.get("name") or (api_path.split(".")[-1] if api_path else "")
      if not name or name.startswith("_"):
        continue

      doc = entry.get("docstring") or ""
      summary = self._clean_docstring(doc)

      params = entry.get("params", [])
      std_args: List[Tuple[str, str]] = []
      posonly_args: List[str] = []
      kwonly_args: List[str] = []

      for p in params:
        if not isinstance(p, dict):
          continue
        p_name = p.get("name", "")
        if not p_name:
          continue
        p_kind = p.get("kind", "")
        p_annot = p.get("annotation") or "Any"

        if p_kind == "POSITIONAL_ONLY":
          posonly_args.append(p_name)
        elif p_kind == "KEYWORD_ONLY":
          kwonly_args.append(p_name)

        if p_kind not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
          std_args.append((p_name, p_annot))

      defn: Dict[str, Any] = {
        "from": f"snapshots/{snapshot_path.name}",
        "description": summary,
        "std_args": std_args,
        "posonly_args": posonly_args,
        "kwonly_args": kwonly_args,
        "returns_type": entry.get("returns_type", "Any"),
        "kind": entry.get("kind", "function"),
      }

      self.validate_function_signature(name, defn)
      semantics[name] = defn

    return semantics

  def validate_function_signature(self, op_name: str, defn: Dict[str, Any]) -> bool:
    """Validate parameter names, standard args, and keyword-only constraints.

    Args:
        op_name: Name of the operator.
        defn: Dictionary containing operator definition and metadata.

    Returns:
        bool: True if signature conforms to Array API standards.

    Raises:
        ValueError: If op_name or parameters are malformed.
    """
    if not op_name or not isinstance(op_name, str):
      raise ValueError("Invalid operation name.")
    std_args = defn.get("std_args", [])
    if not isinstance(std_args, list):
      raise ValueError(f"std_args for {op_name} must be a list.")
    for item in std_args:
      if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError(f"Malformed argument spec in {op_name}: {item}")
      param_name, _param_type = item
      clean_name = param_name.lstrip("*")
      if not clean_name.isidentifier():
        raise ValueError(f"Invalid parameter name '{param_name}' in {op_name}.")
    return True

  def sync_with_snapshot(self, snapshot_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Synchronize semantic definitions against the standard Array API snapshot.

    Args:
        snapshot_path: Optional path to the array API snapshot JSON file.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of validated semantic definitions.
    """
    return self.parse_snapshot(snapshot_path)

  def parse_folder(self, root_dir: Path) -> typing.Dict[str, dict]:
    """Parse Array API Python Stubs (``*.py``) in the target directory.

    Args:
        root_dir: Path to the folder containing .py stubs
                  (e.g. ``src/array_api_stubs/_2023_12``).

    Returns:
        Dict mapping function/constant names to their definitions.

    """
    py_files = list(root_dir.glob("*.py"))

    if not py_files:
      log_warning("No .py files found. Please point to the Python stubs directory (e.g. _2024_12).")
      return {}

    log_info(f"Parsing {len(py_files)} stub files...")
    return self._parse_stubs(py_files, root_dir)

  def _parse_stubs(self, files: typing.List[Path], root: Path) -> typing.Dict[str, dict]:
    """Iterate over files and extracts AST nodes.

    Processes both function definitions and constant assignments (e.g. math constants).

    Args:
        files: List of file paths to parse.
        root: Root directory for relative path calculation.

    Returns:
        A dictionary of parsed semantic definitions.

    """
    semantics: Any = {}

    for fpath in files:
      # Skip internal files (like _types.py), but keep magic methods (__init__ usually re-exports, skip it too)
      if fpath.name.startswith("_") and fpath.name != "__init__.py":
        continue

      # Relative Path: e.g. "_2024_12/elementwise_functions.py"
      # We try to keep the parent folder name for context
      try:
        rel_path = str(fpath.relative_to(root.parent))
      except ValueError:
        rel_path = fpath.name

      try:
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
      except Exception as e:
        log_warning(f"Failed to parse {fpath.name}: {e}")
        continue

      for i, node in enumerate(tree.body):
        # --- CASE 1: FUNCTIONS ---
        if isinstance(node, ast.FunctionDef):
          op_name = node.name
          if op_name.startswith("_") and not op_name.startswith("__"):
            continue  # Skip private helpers

          doc = ast.get_docstring(node)
          summary = self._clean_docstring(doc)

          # Extract typed arguments (name, type_str)
          args = self._extract_args(node.args)

          semantics[op_name] = {
            "from": rel_path,
            "description": summary,
            "std_args": args,
          }

        # --- CASE 2: CONSTANTS (e.g., e = 2.718) ---
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
          name = self._get_assignment_name(node)
          if not name or name.startswith("_"):
            continue

          # Look ahead for Docstring (Expr -> Constant string)
          summary = f"Constant: {name}"
          if i + 1 < len(tree.body):
            next_node = tree.body[i + 1]
            if isinstance(next_node, ast.Expr) and isinstance(next_node.value, ast.Constant):
              if isinstance(next_node.value.value, str):
                summary = self._clean_docstring(next_node.value.value)

          semantics[name] = {
            "from": rel_path,
            "description": summary,
            "std_args": [],  # Constants have no args
            "variants": {},
          }

    return semantics

  def _extract_args(self, args: ast.arguments) -> List[Tuple[str, str]]:
    """Combine Positional-Only, Standard, and Keyword-Only args alongside their type hints.

    Args:
        args: The arguments node from a function definition.

    Returns:
        List of tuples: ``[("x", "Array"), ("axis", "int | None"), ...]``

    """
    out = []

    # Helper to process a specific arg group
    def process_group(group: typing.List[ast.arg]) -> None:
      """Parse type annotations for a list of arguments and appends them to out.

      Args:
          group: A list of AST argument nodes to process.

      Returns:
          None.
      """
      for a in group:
        parsed_type = self._parse_annotation(a.annotation)
        out.append((a.arg, parsed_type))

    # 1. Positional Only ( Python / syntax )
    process_group(args.posonlyargs)
    # 2. Standard
    process_group(args.args)
    # 3. Keyword Only ( * syntax )
    process_group(args.kwonlyargs)

    return out

  def _parse_annotation(self, annotation: Optional[ast.AST]) -> str:
    """Recursively resolves AST type annotations to a readable string representation.

    e.g. ``Name('int')`` -> 'int'
         ``BinOp(Subscript('Optional'), 'int')`` -> 'Optional[int]'  (simplified).

    Args:
        annotation: The AST node representing the type annotation.

    Returns:
        A string representation of the type.

    """
    if annotation is None:
      return "Any"

    if isinstance(annotation, ast.Name):
      return annotation.id

    elif isinstance(annotation, ast.Constant):
      return str(annotation.value)

    elif isinstance(annotation, ast.Subscript):
      # e.g. Optional[int] or Tuple[int, int]
      val = self._parse_annotation(annotation.value)
      if hasattr(annotation, "slice"):
        # Python < 3.9 used ast.Index, 3.9+ uses bare node
        slice_node = annotation.slice
        # Handle Tuples
        if isinstance(slice_node, ast.Tuple):
          dims = [self._parse_annotation(e) for e in slice_node.elts]
          inner = ", ".join(dims)
        else:
          inner = self._parse_annotation(slice_node)
        return f"{val}[{inner}]"
      return val

    elif isinstance(annotation, ast.BinOp):
      # e.g. int | float (Python 3.10+ Union style)
      if isinstance(annotation.op, ast.BitOr):
        left = self._parse_annotation(annotation.left)
        right = self._parse_annotation(annotation.right)
        return f"{left} | {right}"

    elif isinstance(annotation, ast.Attribute):
      # e.g. types.NoneType
      return f"{self._parse_annotation(annotation.value)}.{annotation.attr}"

    return "Any"  # Fallback for complex structures

  def _get_assignment_name(self, node: ast.AST) -> Optional[str]:
    """Extract variable name from Assign (x=1) or AnnAssign (x:int=1).

    Args:
        node: The assignment node.

    Returns:
        The variable name or None.

    """
    if isinstance(node, ast.Assign):
      if node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    elif isinstance(node, ast.AnnAssign):
      if isinstance(node.target, ast.Name):
        return node.target.id
    return None

  def _clean_docstring(self, doc: Optional[str]) -> str:
    """Clean up a docstring to return just the first paragraph summary.

    Args:
        doc: The full docstring.

    Returns:
        A single-line summary string.

    """
    if not doc:
      return ""
    # Take just the first paragraph (up to empty line)
    summary = doc.strip().split("\n\n")[0]
    # Flatten newlines within that paragraph
    return summary.replace("\n", " ").strip()
