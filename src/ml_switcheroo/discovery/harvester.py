"""
Semantic Harvester for extracting knowledge from manual tests.

This module provides the ``SemanticHarvester``, which inspects Python test files
written by developers (Human Override Workflow). It analyzes successful test
calls to reverse-engineer valid argument mappings for the Knowledge Base.

Logic:
    1. Parse a test file's AST.
    2. Scan for imports to resolve aliases (e.g. ``import jax.numpy as jnp``).
    3. Identify test functions (e.g., ``test_matmul``).
    4. Find calls to the target framework (e.g., ``jax.numpy.matmul``).
    5. Correlate input arguments to target parameters using:
       - **Naming Convention**: `np_x` matches standard arg `x`.
       - **Explicit Keywords**: `kwargs` match standard arg names directly.
       - **Value-Based Inference**: Matches literals (e.g. `1`, `True`) to
         type hints defined in the Semantics (e.g., `axis: int`).
    6. Construct and return valid mapping dictionaries.
"""

import ast
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import log_info, log_success, log_warning


class ImportScanner(ast.NodeVisitor):
  """
  Scans AST for imports relevant to the target framework to build an alias map.
  """

  def __init__(self, root_fw: str):
    self.root_fw = root_fw
    # Map of alias -> full_path (e.g. 'jnp' -> 'jax.numpy').
    self.aliases = {}

  def visit_Import(self, node: ast.Import) -> Any:
    for alias in node.names:
      if alias.name.startswith(self.root_fw):
        store_as = alias.asname if alias.asname else alias.name
        self.aliases[store_as] = alias.name
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
    if node.module and node.module.startswith(self.root_fw):
      for alias in node.names:
        full_name = f"{node.module}.{alias.name}"
        store_as = alias.asname if alias.asname else alias.name
        self.aliases[store_as] = full_name
    self.generic_visit(node)


class SemanticHarvester:
  """
  Analyzes Python source code to extract valid API signatures from usage.
  """

  def __init__(self, semantics: SemanticsManager, target_fw: str = "jax"):
    """
    Initializes the harvester.

    Args:
        semantics: Knowledge base manager.
        target_fw: The framework import root to look for (default: jax).
    """
    self.semantics = semantics
    self.target_fw = target_fw

  def harvest_file(self, file_path: Path, dry_run: bool = False) -> int:
    """
    Scans a file, extracts mappings, and updates the semantics JSONs.

    Args:
        file_path: Path to the python test file.
        dry_run: If True, does not write changes to disk.

    Returns:
        int: Number of definitions updated.
    """
    if not file_path.exists():
      log_warning(f"File not found: {file_path}")
      return 0

    try:
      content = file_path.read_text(encoding="utf-8")
      tree = ast.parse(content)
    except Exception as e:
      log_warning(f"Failed to parse {file_path}: {e}")
      return 0

    # 1. Scan Imports at Module Level
    scanner = ImportScanner(self.target_fw)
    scanner.visit(tree)
    aliases = scanner.aliases

    updates = {}

    # 2. Walk top-level functions (tests)
    for node in tree.body:
      if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
        # Heuristic: test_abs -> abs, test_matmul -> matmul
        op_name = self._infer_op_name(node.name)
        if not op_name:
          continue

        # 3. Find valid call signature within this test function
        mapping = self._analyze_test_body(node, op_name, aliases)
        if mapping:
          updates[op_name] = mapping

    # 4. Apply updates
    count = 0
    for op_name, arg_map in updates.items():
      if self._apply_update(op_name, arg_map, dry_run):
        count += 1

    if count > 0 and not dry_run:
      log_success(f"Harvested {count} manual mappings from {file_path.name}")

    return count

  def _infer_op_name(self, test_func_name: str) -> Optional[str]:
    """
    Extracts abstract operation name from test function name.

    Examples:
        'test_matmul' -> 'matmul'
        'test_gen_abs' -> 'abs'

    Args:
        test_func_name: The name of the function def.

    Returns:
        Extracted operation name or None.
    """
    if test_func_name.startswith("test_gen_"):
      return test_func_name[9:]
    if test_func_name.startswith("test_"):
      return test_func_name[5:]
    return None

  def _analyze_test_body(
    self, func_node: ast.FunctionDef, op_name: str, aliases: Dict[str, str]
  ) -> Optional[Dict[str, str]]:
    """
    Inspects function body to find a call to the target API.

    Args:
        func_node: The AST node of the test function.
        op_name: The abstract name of the operation being tested.
        aliases: Map of import aliases resolved in the file.

    Returns:
        A dictionary mapping {std_arg: target_arg} if found, else None.
    """
    defn = self.semantics.get_definition_by_id(op_name)
    if not defn:
      return None

    variants = defn.get("variants", {})
    target_variant = variants.get(self.target_fw)
    if not target_variant or "api" not in target_variant:
      return None

    target_api_path = target_variant["api"]
    # Extract Standard Arguments with Types: [("x", "Array"), ("axis", "int")]
    std_args_info = defn.get("std_args", [])

    visitor = TargetCallVisitor(target_api_path, aliases, std_args_info)
    visitor.visit(func_node)
    return visitor.mappings

  def _apply_update(self, op_name: str, arg_map: Dict[str, str], dry_run: bool) -> bool:
    """
    Updates the SemanticsManager with the harvested mapping.

    Args:
        op_name: Abstract operation identifier.
        arg_map: The harvested argument map.
        dry_run: If True, log only.

    Returns:
        bool: True if an update was staged/committed.
    """
    defn = self.semantics.get_definition_by_id(op_name)
    if not defn:
      return False

    current_data = defn
    variants = current_data.get("variants", {})

    if self.target_fw not in variants:
      return False

    target_variant = variants[self.target_fw]
    if not isinstance(target_variant, dict):
      return False

    # Compare Args
    old_args = target_variant.get("args", {})
    if old_args == arg_map:
      return False  # No change required

    if not dry_run:
      target_variant["args"] = arg_map
      self.semantics.update_definition(op_name, current_data)
    else:
      log_info(f"[Dry Run] Would update {op_name} args to: {arg_map}")

    return True


class TargetCallVisitor(ast.NodeVisitor):
  """
  Helper AST walker to find specific API calls and extract arguments.

  Implements **Value-Based Inference**:
  - Matches naming conventions (``np_x``).
  - Matches literal types (e.g. ``1`` is ``int``) to Semantic Types (``axis: int``).
  """

  def __init__(
    self,
    target_api: str,
    aliases: Dict[str, str],
    std_args_info: List[Any],
  ):
    """
    Initializes the visitor.

    Args:
        target_api: The full path of the function to find.
        aliases: Import aliases map.
        std_args_info: Standard argument definitions from Spec.
    """
    self.target_api = target_api
    self.aliases = aliases
    # Normalize std_args to [(name, type), ...] format
    self.std_args_info = self._normalize_info(std_args_info)
    self.mappings: Optional[Dict[str, str]] = None

  def _normalize_info(self, raw: List[Any]) -> List[Tuple[str, str]]:
    """Ensures consistent list of tuples structure."""
    out = []
    for item in raw:
      if isinstance(item, (list, tuple)) and len(item) >= 2:
        out.append((item[0], item[1]))
      elif isinstance(item, str):
        out.append((item, "Any"))
    return out

  def visit_Call(self, node: ast.Call) -> Any:
    """
    Inspects calls. Check if they match target_api and extract args.
    """
    if self.mappings is not None:
      return

    call_name = self._resolve_call_name(node.func)
    if not call_name:
      return

    if call_name != self.target_api:
      return

    extracted_map = {}

    # 1. Process Keyword Arguments with Enhanced Inference
    for kw in node.keywords:
      tgt_arg = kw.arg
      if not tgt_arg:
        continue

      # Heuristic A: Variable Name Convention (np_x -> x)
      val_name = self._get_arg_val_name(kw.value)
      if val_name:
        std_arg = self._clean_std_name(val_name)
        # Verify std_arg exists in spec? ideally yes, but loose match ok
        extracted_map[std_arg] = tgt_arg
        continue

      # Heuristic B: Value-Based Inference via Literal Matching
      # e.g. call(axis=1) matches std_arg 'axis' type 'int'
      val_type = self._infer_literal_type(kw.value)
      if val_type != "Any":
        match = self._find_std_arg_by_type(val_type)
        if match:
          extracted_map[match] = tgt_arg
          continue

      # Heuristic C: Implicit Keyword Match
      # If literal matching failed or was ambiguous, assume kw arg is std arg
      extracted_map[tgt_arg] = tgt_arg

    if extracted_map:
      self.mappings = extracted_map

    self.generic_visit(node)

  def _resolve_call_name(self, node: ast.AST) -> str:
    """Resolves AST Attribute/Name to full dotted string using aliases."""
    parts = []
    curr = node
    while isinstance(curr, ast.Attribute):
      parts.insert(0, curr.attr)
      curr = curr.value

    if isinstance(curr, ast.Name):
      root_name = curr.id
      real_root = self.aliases.get(root_name, root_name)
      parts.insert(0, real_root)
      return ".".join(parts)

    return ""

  def _get_arg_val_name(self, node: ast.AST) -> Optional[str]:
    """Extracts variable name from AST node if it is a Name."""
    if isinstance(node, ast.Name):
      return node.id
    return None

  def _clean_std_name(self, var_name: str) -> str:
    """Converts variable name to abstract standard name."""
    if var_name.startswith("np_"):
      return var_name[3:]
    return var_name

  def _infer_literal_type(self, node: ast.AST) -> str:
    """
    Determines logical type of a literal node.

    Args:
        node: AST node.

    Returns:
        String: 'int', 'float', 'bool', 'str', or 'Any'.
    """
    if isinstance(node, ast.Constant):
      val = node.value
      if isinstance(val, bool):
        return "bool"
      if isinstance(val, int):
        return "int"
      if isinstance(val, float):
        return "float"
      if isinstance(val, str):
        return "str"
    return "Any"

  def _find_std_arg_by_type(self, val_type: str) -> Optional[str]:
    """
    Constraints Solver: Find unique std_arg that matches this type.

    If multiple args match (e.g. two ints), returns None (Ambiguous).
    """
    candidates = []
    for name, type_hint in self.std_args_info:
      # Check simple match (hint usually 'int' or 'Optional[int]')
      if val_type in type_hint or type_hint == "Any":
        candidates.append(name)

    if len(candidates) == 1:
      return candidates[0]

    # Refined check for specific common names if ambiguous
    # e.g. if we have axis(int) and other(int), prioritize axis
    priority = ["axis", "dim", "keepdims"]
    for p in priority:
      if p in candidates:
        return p

    return None
