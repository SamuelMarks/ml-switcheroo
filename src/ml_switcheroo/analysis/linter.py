"""Structural Validation Linter (The Anti-Pollution Check).

Provides ``StructuralLinter`` and ``validate_transpilation``, static analysis utilities
verifying that transpiled output contains no residual imports, aliases, or attributes
from the source framework.
"""

import libcst as cst


class StructuralLinter(cst.CSTVisitor):
  """Scan CST for forbidden framework usage.

  Attributes:
      forbidden_roots: Root package names forbidden in transpiled output (`set[str]`).
      violations: Accumulated violation messages (`list[str]`).
  """

  def __init__(self, forbidden_roots: set[str]) -> None:
    """Initialize the linter.

    Args:
        forbidden_roots: Set of root packages to ban (e.g. ``{"torch"}``).
    """
    self.forbidden_roots: set[str] = forbidden_roots
    self.violations: list[str] = []
    self._local_aliases: dict[str, str] = {}
    self._context_stack: list[str] = []

  def check(self, code: str) -> list[str]:
    """Run the linter on a source string.

    Args:
        code: Python source code to validate.

    Returns:
        list[str]: Collected error messages (empty if valid).
    """
    self.violations = []
    self._local_aliases = {}
    self._context_stack = []

    try:
      tree = cst.parse_module(code)
      tree.visit(self)
    except (cst.ParserSyntaxError, ValueError, TypeError, AttributeError, RuntimeError) as e:
      self.violations.append(f"Linter Parse Error: {e}")

    return self.violations

  def visit_Import(self, node: cst.Import) -> None:
    """Check ``import x`` and ``import x as y`` statements.

    Args:
        node: Import CST node.
    """
    self._context_stack.append("import")
    for alias in node.names:
      root = self._get_root_name(alias.name)
      if root in self.forbidden_roots:
        self.violations.append(f"Forbidden Import: '{root}'")
        local_name = (
          (alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else "") if alias.asname else root
        )
        self._local_aliases[local_name] = root

  def leave_Import(self, node: cst.Import) -> None:
    """Exit the import scope context.

    Args:
        node: Import CST node being exited.
    """
    self._context_stack.pop()

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Check ``from x import y`` statements.

    Args:
        node: ImportFrom CST node.
    """
    self._context_stack.append("import")
    if node.module:
      module_name = self._get_full_name_from_node(node.module)
      root = module_name.split(".")[0]
      if root in self.forbidden_roots:
        self.violations.append(f"Forbidden Import: 'from {module_name} ...'")
        if isinstance(node.names, cst.ImportStar):
          self.violations.append(f"Forbidden Wildcard Import from '{root}'")
        else:
          for alias in list(node.names):
            if isinstance(alias, cst.ImportAlias):
              local_name = (
                (alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else "")
                if alias.asname
                else (alias.name.value if isinstance(alias.name, cst.Name) else "")
              )
              self._local_aliases[local_name] = root

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Exit the import-from scope context.

    Args:
        node: ImportFrom CST node being exited.
    """
    self._context_stack.pop()

  def visit_Name(self, node: cst.Name) -> None:
    """Check usage of aliased or direct forbidden package names outside import statements.

    Args:
        node: Name CST node.
    """
    if "import" in self._context_stack:
      return
    if node.value in self._local_aliases:
      root = self._local_aliases[node.value]
      msg = f"Forbidden Usage: Alias '{node.value}' (alias of {root})"
      if msg not in self.violations:
        self.violations.append(msg)
    elif node.value in self.forbidden_roots:
      msg = f"Forbidden Usage: Direct access '{node.value}'"
      if msg not in self.violations:
        self.violations.append(msg)

  def visit_Attribute(self, node: cst.Attribute) -> None:
    """Check qualified attribute accesses on forbidden aliases or roots.

    Args:
        node: Attribute CST node.
    """
    if isinstance(node.value, cst.Name):
      name = node.value.value
      if name in self._local_aliases:
        root = self._local_aliases[name]
        msg = f"Forbidden Attribute: '{name}.{node.attr.value}' (alias of {root})"
        if msg not in self.violations:
          self.violations.append(msg)
      elif name in self.forbidden_roots:
        msg = f"Forbidden Attribute: Direct root '{name}.{node.attr.value}'"
        if msg not in self.violations:
          self.violations.append(msg)

  def _get_root_name(self, node: cst.BaseExpression) -> str:
    """Extract root package from dotted path node.

    Args:
        node: Expression CST node.

    Returns:
        str: Root package string.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return self._get_root_name(node.value)
    return ""

  def _get_full_name_from_node(self, node: cst.BaseExpression) -> str:
    """Recursively resolve CST node to a dot-separated string.

    Args:
        node: Expression CST node.

    Returns:
        str: Full dot-separated string name.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      return f"{self._get_full_name_from_node(node.value)}.{node.attr.value}"
    return ""


__all__ = ["StructuralLinter"]
