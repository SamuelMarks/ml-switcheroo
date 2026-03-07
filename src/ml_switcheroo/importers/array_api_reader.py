"""
Importer for Python Array API Standard Stubs.

This module parses .py stub files from the official Array API standard repo,
extracting function signatures, type hints, and docstrings to build the
Semantic Knowledge Base.

Feature 027 Update:
Now extracts type hints (e.g. ``x: Array``, ``axis: int``) to support Better Fuzzing.
"""

import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from ml_switcheroo.utils.console import log_info, log_warning


class ArrayApiSpecImporter:
  """
  Parses Python stub files (``*.py``) using the built-in ``ast`` module.
  """

  def parse_folder(self, root_dir: Path) -> Dict[str, Any]:
    """
    Parses Array API Python Stubs (``*.py``) in the target directory.

    Args:
        root_dir: Path to the folder containing .py stubs
                  (e.g. ``src/array_api_stubs/_2023_12``).

    Returns:
        Dict mapping function/constant names to their definitions.
    """
    py_files = list(root_dir.glob("*.py"))  # pragma: no cover

    if not py_files:  # pragma: no cover
      log_warning("No .py files found. Please point to the Python stubs directory (e.g. _2024_12).")  # pragma: no cover
      return {}  # pragma: no cover

    log_info(f"Parsing {len(py_files)} stub files...")  # pragma: no cover
    return self._parse_stubs(py_files, root_dir)  # pragma: no cover

  def _parse_stubs(self, files: List[Path], root: Path) -> Dict[str, Any]:
    """
    Iterates over files and extracts AST nodes.

    Processes both function definitions and constant assignments (e.g. math constants).

    Args:
        files: List of file paths to parse.
        root: Root directory for relative path calculation.

    Returns:
        A dictionary of parsed semantic definitions.
    """
    semantics = {}

    for fpath in files:
      # Skip internal files (like _types.py), but keep magic methods (__init__ usually re-exports, skip it too)
      if fpath.name.startswith("_") and fpath.name != "__init__.py":
        continue  # pragma: no cover

      # Relative Path: e.g. "_2024_12/elementwise_functions.py"
      # We try to keep the parent folder name for context
      try:
        rel_path = str(fpath.relative_to(root.parent))
      except ValueError:  # pragma: no cover
        rel_path = fpath.name  # pragma: no cover

      try:
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
      except Exception as e:  # pragma: no cover
        log_warning(f"Failed to parse {fpath.name}: {e}")  # pragma: no cover
        continue  # pragma: no cover

      for i, node in enumerate(tree.body):
        # --- CASE 1: FUNCTIONS ---
        if isinstance(node, ast.FunctionDef):
          op_name = node.name
          if op_name.startswith("_") and not op_name.startswith("__"):
            continue  # Skip private helpers  # pragma: no cover

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
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):  # pragma: no cover
          name = self._get_assignment_name(node)  # pragma: no cover
          if not name or name.startswith("_"):  # pragma: no cover
            continue  # pragma: no cover

          # Look ahead for Docstring (Expr -> Constant string)
          summary = f"Constant: {name}"  # pragma: no cover
          if i + 1 < len(tree.body):  # pragma: no cover
            next_node = tree.body[i + 1]  # pragma: no cover
            if isinstance(next_node, ast.Expr) and isinstance(next_node.value, ast.Constant):  # pragma: no cover
              if isinstance(next_node.value.value, str):  # pragma: no cover
                summary = self._clean_docstring(next_node.value.value)  # pragma: no cover

          semantics[name] = {  # pragma: no cover
            "from": rel_path,
            "description": summary,
            "std_args": [],  # Constants have no args
            "variants": {},
          }

    return semantics

  def _extract_args(self, args: ast.arguments) -> List[Tuple[str, str]]:
    """
    Combines Positional-Only, Standard, and Keyword-Only args alongside their type hints.

    Args:
        args: The arguments node from a function definition.

    Returns:
        List of tuples: ``[("x", "Array"), ("axis", "int | None"), ...]``
    """
    out = []

    # Helper to process a specific arg group
    def process_group(group: List[ast.arg]):
      """TODO: Add docstring."""
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
    """
    Recursively resolves AST type annotations to a readable string string.
    e.g. ``Name('int')`` -> 'int'
         ``BinOp(Subscript('Optional'), 'int')`` -> 'Optional[int]'  (simplified)

    Args:
        annotation: The AST node representing the type annotation.

    Returns:
        A string representation of the type.
    """
    if annotation is None:
      return "Any"

    if isinstance(annotation, ast.Name):
      return annotation.id

    elif isinstance(annotation, ast.Constant):  # pragma: no cover
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
        return f"{val}[{inner}]"  # pragma: no cover
      return val  # pragma: no cover

    elif isinstance(annotation, ast.BinOp):
      # e.g. int | float (Python 3.10+ Union style)
      if isinstance(annotation.op, ast.BitOr):
        left = self._parse_annotation(annotation.left)
        right = self._parse_annotation(annotation.right)
        return f"{left} | {right}"
    # pragma: no cover
    elif isinstance(annotation, ast.Attribute):  # pragma: no cover
      # e.g. types.NoneType  # pragma: no cover
      return f"{self._parse_annotation(annotation.value)}.{annotation.attr}"  # pragma: no cover
    # pragma: no cover
    return "Any"  # Fallback for complex structures  # pragma: no cover

  def _get_assignment_name(self, node: ast.AST) -> Optional[str]:
    """
    Extracts variable name from Assign (x=1) or AnnAssign (x:int=1).

    Args:
        node: The assignment node.

    Returns:
        The variable name or None.
    """  # pragma: no cover
    if isinstance(node, ast.Assign):  # pragma: no cover
      if node.targets and isinstance(node.targets[0], ast.Name):  # pragma: no cover
        return node.targets[0].id  # pragma: no cover
    elif isinstance(node, ast.AnnAssign):  # pragma: no cover
      if isinstance(node.target, ast.Name):  # pragma: no cover
        return node.target.id  # pragma: no cover
    return None  # pragma: no cover

  def _clean_docstring(self, doc: Optional[str]) -> str:
    """
    Cleans up a docstring return just the first paragraph summary.

    Args:
        doc: The full docstring.

    Returns:
        A single-line summary string.
    """
    if not doc:
      return ""
    # Take just the first paragraph (up to empty line)  # pragma: no cover
    summary = doc.strip().split("\n\n")[0]  # pragma: no cover
    # Flatten newlines within that paragraph  # pragma: no cover
    return summary.replace("\n", " ").strip()  # pragma: no cover
