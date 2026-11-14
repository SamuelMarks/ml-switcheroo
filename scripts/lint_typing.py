"""AST-based static typing and anti-reflection linter for ml-switcheroo.

Enforces zero usage of:
- ``getattr``, ``setattr``, ``hasattr``, ``delattr``, ``vars``, ``globals``, ``locals``, ``eval``, ``exec``, ``compile``, ``__import__``, ``cast``
- ``object`` and ``Hashable`` (as catch-all type annotations, base classes, or builtin references)
- ``operator.attrgetter``, ``operator.itemgetter``, ``operator.methodcaller``
- ``object.__setattr__``, ``object.__getattribute__``, ``object.__delattr__``, ``.__dict__``, ``.__mro__``, ``.__subclasses__``, ``.__class__``
- Direct dunder method invocations ``obj.__<name>__(...)`` (only ``super().__<name>__(...)`` is permitted)
- Dynamic ``def __getattr__`` / ``async def __getattr__`` definitions
- ``typing.Any``, ``typing_extensions.Any``, ``typing.Hashable``, ``collections.abc.Hashable``
- ``typing.cast``, ``typing_extensions.cast``
- Module or symbol import aliases attempting to disguise any of the above
- ``# type: ignore`` comments
"""

import ast
import io
import sys
import tokenize
from pathlib import Path
from typing import Sequence

_FORBIDDEN_BUILTIN_CALLS: frozenset[str] = frozenset(
  {
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "vars",
    "globals",
    "locals",
    "eval",
    "exec",
    "compile",
    "__import__",
    "cast",
    "object",
  }
)

_FORBIDDEN_OPERATOR_NAMES: frozenset[str] = frozenset(
  {
    "attrgetter",
    "itemgetter",
    "methodcaller",
  }
)

_OPERATOR_MODULES: frozenset[str] = frozenset({"operator", "_operator"})

_FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
  {
    "__dict__",
    "__mro__",
    "__subclasses__",
    "__setattr__",
    "__getattribute__",
    "__delattr__",
    "__class__",
  }
)

_FORBIDDEN_DUNDER_DEFS: frozenset[str] = frozenset(
  {
    "__getattr__",
    "__getattribute__",
  }
)

_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions", "collections.abc"})
_FORBIDDEN_TYPING_NAMES: frozenset[str] = frozenset({"Any", "cast", "Hashable"})
_FORBIDDEN_BARE_NAMES: frozenset[str] = frozenset({"Any", "object", "Hashable"})
_MOCK_MODULES: frozenset[str] = frozenset({"unittest.mock", "mock"})


def _is_production_path(file_path: Path) -> bool:
  """Check whether ``file_path`` resides under production ``src/``, ``scripts/``, or ``docs/`` trees.

  Args:
      file_path: Candidate Python source file path.

  Returns:
      True when ``file_path`` belongs to ``src``, ``scripts``, or ``docs`` (excluding ``tests``).
  """
  parts = file_path.parts
  return ("src" in parts or "scripts" in parts or "docs" in parts) and "tests" not in parts


class _TypingAndReflectionVisitor(ast.NodeVisitor):
  """AST visitor collecting static typing, anti-reflection, and anti-dunder violations.

  Attributes:
      file_path: Path of the Python source file being audited.
      is_production: True if ``file_path`` is inside production ``src/`` or ``scripts/``.
      violations: Accumulated formatted lint violation strings.
      module_aliases: Mapping from local alias identifier to canonical imported module path.
      forbidden_aliases: Set of local identifiers bound to forbidden imported symbols.
  """

  def __init__(self, file_path: Path) -> None:
    """Initialize the visitor.

    Args:
        file_path: Path of the file being checked.
    """
    self.file_path: Path = file_path
    self.is_production: bool = _is_production_path(file_path)
    self.violations: list[str] = []
    self.module_aliases: dict[str, str] = {}
    self.forbidden_aliases: set[str] = set()

  def _record(self, lineno: int, col_offset: int, message: str) -> None:
    """Record a lint violation.

    Args:
        lineno: 1-based line number.
        col_offset: 0-based column offset.
        message: Human-readable violation description.
    """
    self.violations.append(f"{self.file_path}:{lineno}:{col_offset + 1}: {message}")

  def _check_func_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Check sync or async function definitions for forbidden dynamic dunders.

    Args:
        node: Sync or async function AST definition node.
    """
    if node.name in _FORBIDDEN_DUNDER_DEFS:
      prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
      self._record(
        node.lineno,
        node.col_offset,
        f"Forbidden dynamic method definition '{prefix} {node.name}'",
      )
    self.generic_visit(node)

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    """Check synchronous function definitions for ``__getattr__`` / ``__getattribute__``.

    Args:
        node: FunctionDef AST node.
    """
    self._check_func_def(node)

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
    """Check asynchronous function definitions for ``__getattr__`` / ``__getattribute__``.

    Args:
        node: AsyncFunctionDef AST node.
    """
    self._check_func_def(node)

  def visit_Import(self, node: ast.Import) -> None:
    """Record module aliases (`import typing as t`, `import builtins as b`, `import collections.abc as cabc`).

    Args:
        node: Import AST node.
    """
    for alias in node.names:
      local_name = alias.asname or alias.name.split(".")[0]
      self.module_aliases[local_name] = alias.name
      if self.is_production and (alias.name in _MOCK_MODULES or alias.name.startswith("unittest.mock.")):
        self._record(node.lineno, node.col_offset, f"Forbidden mock import '{alias.name}' in production code")
    self.generic_visit(node)

  def _check_production_import_from(self, node: ast.ImportFrom, module: str) -> None:
    """Enforce production-only import restrictions (no ``unittest.mock``).

    Args:
        node: ImportFrom AST node.
        module: Resolved source module name.
    """
    if module in _MOCK_MODULES or module.startswith("unittest.mock."):
      self._record(node.lineno, node.col_offset, f"Forbidden mock import from '{module}' in production code")
    elif module == "unittest":
      for alias in node.names:
        if alias.name == "mock":
          self._record(node.lineno, node.col_offset, "Forbidden mock import 'unittest.mock' in production code")

  def _record_forbidden_aliases(
    self,
    node: ast.ImportFrom,
    prefix: str,
    forbidden_names: frozenset[str],
  ) -> None:
    """Record violations and aliases for forbidden imported names from ``prefix``.

    Args:
        node: ImportFrom AST node.
        prefix: Canonical module name used in diagnostic messages.
        forbidden_names: Set of forbidden symbol names in ``prefix``.
    """
    for alias in node.names:
      if alias.name in forbidden_names:
        self._record(node.lineno, node.col_offset, f"Forbidden import '{prefix}.{alias.name}'")
        self.forbidden_aliases.add(alias.asname or alias.name)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    """Check forbidden symbol imports and track module/symbol aliases (`from collections import abc as c`).

    Args:
        node: ImportFrom AST node.
    """
    module = node.module or ""
    if self.is_production:
      self._check_production_import_from(node, module)
    if module in _TYPING_MODULES:
      self._record_forbidden_aliases(node, module, _FORBIDDEN_TYPING_NAMES)
    elif module == "collections":
      for alias in node.names:
        if alias.name == "abc":
          self.module_aliases[alias.asname or "abc"] = "collections.abc"
    elif module == "builtins":
      self._record_forbidden_aliases(node, "builtins", _FORBIDDEN_BUILTIN_CALLS)
    elif module in _OPERATOR_MODULES:
      self._record_forbidden_aliases(node, module, _FORBIDDEN_OPERATOR_NAMES)
    elif module == "inspect":
      self._record_forbidden_aliases(node, "inspect", frozenset({"getattr_static"}))
    self.generic_visit(node)

  def _resolve_module_expr(self, expr: ast.expr) -> str | None:
    """Resolve `expr` (e.g., `t`, `collections.abc`) to its canonical module name if known.

    Args:
        expr: AST expression node on the left-hand side of an attribute access.

    Returns:
        Canonical module path string if recognized, otherwise `None`.
    """
    if isinstance(expr, ast.Name):
      return self.module_aliases.get(expr.id, expr.id)
    if isinstance(expr, ast.Attribute):
      parent = self._resolve_module_expr(expr.value)
      if parent is not None:
        return f"{parent}.{expr.attr}"
    return None

  def visit_Attribute(self, node: ast.Attribute) -> None:
    """Check forbidden typing/builtin/operator references (including aliases) and dunder attributes.

    Args:
        node: Attribute AST node.
    """
    if node.attr in _FORBIDDEN_DUNDER_ATTRS:
      is_super_dunder = (
        node.attr in ("__setattr__", "__getattribute__", "__delattr__")
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "super"
      )
      if not is_super_dunder:
        self._record(node.lineno, node.col_offset, f"Forbidden dunder attribute access '.{node.attr}'")
    else:
      resolved_mod = self._resolve_module_expr(node.value)
      if resolved_mod in _TYPING_MODULES and node.attr in _FORBIDDEN_TYPING_NAMES:
        self._record(node.lineno, node.col_offset, f"Forbidden reference '{resolved_mod}.{node.attr}'")
      elif resolved_mod == "builtins" and node.attr in _FORBIDDEN_BUILTIN_CALLS:
        self._record(node.lineno, node.col_offset, f"Forbidden builtin reference 'builtins.{node.attr}'")
      elif resolved_mod in _OPERATOR_MODULES and node.attr in _FORBIDDEN_OPERATOR_NAMES:
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden operator reflection reference '{resolved_mod}.{node.attr}'",
        )
      elif resolved_mod == "inspect" and node.attr == "getattr_static":
        self._record(node.lineno, node.col_offset, "Forbidden reflection reference 'inspect.getattr_static'")
    self.generic_visit(node)

  def visit_Call(self, node: ast.Call) -> None:
    """Check direct calls to forbidden builtins, operator helpers, and direct dunder method invocations.

    Args:
        node: Call AST node.
    """
    if isinstance(node.func, ast.Name):
      if (
        node.func.id in _FORBIDDEN_BUILTIN_CALLS
        or node.func.id in _FORBIDDEN_OPERATOR_NAMES
        or node.func.id in self.forbidden_aliases
      ):
        self._record(node.lineno, node.col_offset, f"Forbidden reflection/cast call '{node.func.id}()'")
    elif isinstance(node.func, ast.Attribute):
      attr = node.func.attr
      if len(attr) > 4 and attr.startswith("__") and attr.endswith("__"):
        is_super = (
          isinstance(node.func.value, ast.Call)
          and isinstance(node.func.value.func, ast.Name)
          and node.func.value.func.id == "super"
        )
        if not is_super:
          self._record(node.lineno, node.col_offset, f"Forbidden direct dunder method call '.{attr}()'")
    self.generic_visit(node)

  def visit_Name(self, node: ast.Name) -> None:
    """Check bare ``Any``, ``object``, ``Hashable``, and aliased forbidden symbol references.

    Args:
        node: Name AST node.
    """
    if node.id in _FORBIDDEN_BARE_NAMES or node.id in self.forbidden_aliases:
      self._record(node.lineno, node.col_offset, f"Forbidden reference to '{node.id}'")
    self.generic_visit(node)


def check_source(source: str, file_path: Path) -> list[str]:
  """Check a single Python source string for typing and reflection violations.

  Args:
      source: Python source code text.
      file_path: Path associated with the source text.

  Returns:
      List of formatted violation messages.
  """
  tree = ast.parse(source, filename=str(file_path))
  visitor = _TypingAndReflectionVisitor(file_path)
  visitor.visit(tree)

  tokens = tokenize.generate_tokens(io.StringIO(source).readline)
  for tok in tokens:
    if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
      visitor._record(tok.start[0], tok.start[1], "Forbidden '# type: ignore' comment")

  return visitor.violations


def collect_python_files(paths: Sequence[str | Path]) -> list[Path]:
  """Resolve file and directory arguments into sorted Python file paths.

  Args:
      paths: Sequence of file or directory path strings or Path objects.

  Returns:
      Sorted list of ``.py`` files.
  """
  collected: list[Path] = []
  for raw in paths:
    path = Path(raw)
    if path.is_file() and path.suffix == ".py":
      collected.append(path)
    elif path.is_dir():
      collected.extend(sorted(path.rglob("*.py")))
  return sorted(set(collected))


def main(argv: Sequence[str] | None = None) -> int:
  """Run the AST typing and anti-reflection linter across target paths.

  Args:
      argv: Command-line arguments (defaults to ``src``, ``scripts``, ``docs``, ``tests``).

  Returns:
      Exit code (0 if clean, 1 if violations found).
  """
  args = list(argv) if argv is not None else sys.argv[1:]
  targets = args if args else ["src", "scripts", "docs", "tests"]
  files = collect_python_files(targets)

  all_violations: list[str] = []
  for py_file in files:
    source = py_file.read_text(encoding="utf-8")
    all_violations.extend(check_source(source, py_file))

  if all_violations:
    for violation in all_violations:
      print(violation, file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
