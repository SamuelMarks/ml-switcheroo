"""
Importer for Python Array API Standard Stubs.

This module parses .py stub files from the official Array API standard repo,
extracting function signatures, type hints, and docstrings to build the
Semantic Knowledge Base.

Feature 027 Update:
Now extracts type hints (e.g. 'x: Array', 'axis: int') to support Better Fuzzing.
"""

import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from ml_switcheroo.utils.console import log_info, log_warning


class ArrayApiSpecImporter:
  """
  Parses Python stub files (*.py) using the built-in `ast` module.
  """

  def parse_folder(self, root_dir: Path) -> Dict[str, Any]:
    """
    Parses Array API Python Stubs (*.py) in the target directory.

    Args:
        root_dir: Path to the folder containing .py stubs (e.g. `src/array_api_stubs/_2023_12`).

    Returns:
        Dict mapping function/constant names to their definitions.
    """
    py_files = list(root_dir.glob("*.py"))

    if not py_files:
      log_warning("No .py files found. Please point to the Python stubs directory (e.g. _2024_12).")
      return {}

    log_info(f"Parsing {len(py_files)} stub files...")
    return self._parse_stubs(py_files, root_dir)

  def _parse_stubs(self, files: List[Path], root: Path) -> Dict[str, Any]:
    """
    Iterates over files and extracts AST nodes.
    """
    semantics = {}

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

          semantics[op_name] = {"from": rel_path, "description": summary, "std_args": args, "variants": {}}

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
    """
    Combines Positional-Only, Standard, and Keyword-Only args alongside their type hints.

    Returns:
        List of tuples: [("x", "Array"), ("axis", "int | None"), ...]
    """
    out = []

    # Helper to process a specific arg group
    def process_group(group: List[ast.arg]):
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
    e.g. Name('int') -> 'int'
         BinOp(Subscript('Optional'), 'int') -> 'Optional[int]'  (simplified)
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
    """Extracts variable name from Assign (x=1) or AnnAssign (x:int=1)"""
    if isinstance(node, ast.Assign):
      if node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    elif isinstance(node, ast.AnnAssign):
      if isinstance(node.target, ast.Name):
        return node.target.id
    return None

  def _clean_docstring(self, doc: Optional[str]) -> str:
    if not doc:
      return ""
    # Take just the first paragraph (up to empty line)
    summary = doc.strip().split("\n\n")[0]
    # Flatten newlines within that paragraph
    return summary.replace("\n", " ").strip()
