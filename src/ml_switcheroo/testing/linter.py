"""
Structural Validation Linter (The Anti-Pollution Check).

This module provides the ``StructuralLinter``, a static analysis tool designed
to verify that the output of a transpilation contains no artifacts from the
source framework.

Detection Scope:
1.  **Direct Imports**: Flags `import torch` or `from flax import ...`.
2.  **Aliased Usage**: Tracks `import torch as t` and flags subsequent calls like `t.abs()`.
3.  **Attribute Access**: Flags `torch.nn.Linear` if `torch` is a forbidden root.
"""

import libcst as cst
from typing import List, Set, Dict, Tuple, Optional

from ml_switcheroo.frameworks import get_adapter


class StructuralLinter(cst.CSTVisitor):
  """
  Scans CST for forbidden framework usage.
  """

  def __init__(self, forbidden_roots: Set[str]):
    """
    Initializes the linter.

    Args:
        forbidden_roots: A set of string root packages to ban (e.g. {"torch"}).
    """
    self.forbidden_roots = forbidden_roots
    self.violations: List[str] = []
    # Track local aliases to catch aliased usage (import torch as t; t.abs)
    self._local_aliases: Dict[str, str] = {}
    # Track if we are inside an import statement to handle definition vs usage
    self._context_stack: List[str] = []

  def check(self, code: str) -> List[str]:
    """
    Runs the linter on a source string.

    Args:
        code: The Python source code to validate.

    Returns:
        List[str]: A list of error messages. Empty if valid.
    """
    self.violations = []
    self._local_aliases = {}
    self._context_stack = []

    try:
      tree = cst.parse_module(code)
      tree.visit(self)
    except Exception as e:
      self.violations.append(f"Linter Parse Error: {e}")

    return self.violations

  def visit_Import(self, node: cst.Import) -> None:
    """
    Checks `import x`, `import x as y`.

    Logs violations if the root package is forbidden.
    Tracks aliases to detect usage later in the file.

    Args:
        node: The Import node.
    """
    self._context_stack.append("import")
    for alias in node.names:
      root = self._get_root_name(alias.name)

      if root in self.forbidden_roots:
        self.violations.append(f"Forbidden Import: '{root}'")

        # Track alias for usage detection
        # import torch -> alias 'torch', import torch as t -> alias 't'
        local_name = alias.asname.name.value if alias.asname else root
        self._local_aliases[local_name] = root

  def leave_Import(self, node: cst.Import) -> None:
    """
    Exits the import scope context.

    Args:
        node: The Import node being exited.
    """
    self._context_stack.pop()

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Checks `from x import y`.

    Logs violations if the module root matches forbidden set.

    Args:
        node: The ImportFrom node.
    """
    self._context_stack.append("import")

    # Determine module being imported from
    if node.module:
      module_name = self._get_full_name_from_node(node.module)
      root = module_name.split(".")[0]

      if root in self.forbidden_roots:
        self.violations.append(f"Forbidden Import: 'from {module_name} ...'")

        # Track imported names
        if isinstance(node.names, cst.ImportStar):
          # Wildcard import from forbidden module contaminates global namespace blindly
          self.violations.append(f"Forbidden Wildcard Import from '{root}'")
        else:
          for alias in node.names:
            if isinstance(alias, cst.ImportAlias):
              local_name = alias.asname.name.value if alias.asname else alias.name.value
              self._local_aliases[local_name] = root

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Exits the import-from scope context.

    Args:
        node: The ImportFrom node being exited.
    """
    self._context_stack.pop()

  def visit_Name(self, node: cst.Name) -> None:
    """
    Checks usage of aliased forbidden variables (e.g. `t.abs()` where `t` is torch).

    Ignores names inside import definition statements.

    Args:
        node: The Name node.
    """
    # Iterate check only if we are not defining the import itself
    if "import" in self._context_stack:
      return

    if node.value in self._local_aliases:
      root = self._local_aliases[node.value]
      # Format explicitly matches existing test expectations "alias of {root}"
      msg = f"Forbidden Usage: Alias '{node.value}' (alias of {root})"
      if msg not in self.violations:
        self.violations.append(msg)

  def visit_Attribute(self, node: cst.Attribute) -> None:
    """
    Checks attributes to provide more specific error messages (e.g. `torch.abs`).

    If the left side of the attribute matches a forbidden alias, logs an error.

    Args:
        node: The Attribute node.
    """
    # If the value (left side) of attribute is a forbidden alias
    if isinstance(node.value, cst.Name):
      name = node.value.value
      if name in self._local_aliases:
        root = self._local_aliases[name]
        # Format explicitly matches existing test expectations "alias of {root}"
        msg = f"Forbidden Attribute: '{name}.{node.attr.value}' (alias of {root})"
        if msg not in self.violations:
          self.violations.append(msg)

  def _get_root_name(self, node: cst.BaseExpression) -> str:
    """
    Extracts root package from dotted path node (e.g. 'torch' from 'torch.nn').

    Args:
        node: CST expression node (Name or Attribute).

    Returns:
        str: The root identifier.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return self._get_root_name(node.value)
    return ""

  def _get_full_name_from_node(self, node: cst.BaseExpression) -> str:
    """
    Recursively resolves CST node to dot-separated string.

    Args:
        node: CST expression node.

    Returns:
        str: The full path (e.g. "torch.nn.functional").
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return f"{self._get_full_name_from_node(node.value)}.{node.attr.value}"
    return ""


def validate_transpilation(code: str, source_fw: str) -> Tuple[bool, List[str]]:
  """
  Facade to lint generated code against a specific source framework.

  It automatically expands the `source_fw` into a set of forbidden roots including:
  1. The framework itself (e.g. 'torch').
  2. Known aliases (from adapter).
  3. Inherited parents (e.g. 'flax_nnx' bans 'jax' too).

  Args:
      code: The generated python code.
      source_fw: The framework that SHOULD have been removed (e.g., "torch").

  Returns:
      (is_valid, list_of_errors)
  """
  forbidden = {source_fw}

  adapter = get_adapter(source_fw)
  if adapter:
    # 1. Add primary import alias (e.g. 'flax.nnx' -> 'flax')
    if hasattr(adapter, "import_alias") and adapter.import_alias:
      mod, _ = adapter.import_alias
      forbidden.add(mod.split(".")[0])

    # 2. Add Search Module roots
    if hasattr(adapter, "search_modules"):
      for mod in adapter.search_modules:
        forbidden.add(mod.split(".")[0])

    # 3. Add inheritance parents
    # If converting FROM Flax, we likely want to ban JAX primitives too
    if hasattr(adapter, "inherits_from") and adapter.inherits_from:
      forbidden.add(adapter.inherits_from)

  linter = StructuralLinter(forbidden)
  errors = linter.check(code)

  return len(errors) == 0, errors
