"""AST-based static typing and anti-reflection linter for ml-switcheroo.

Enforces zero usage of:
- ``getattr``, ``setattr``, ``hasattr``, ``delattr``, ``vars``, ``globals``, ``locals``, ``eval``, ``exec``, ``compile``, ``__import__``, ``cast``
- ``object`` (as a type annotation, base class, or builtin reference/call)
- ``operator.attrgetter``, ``operator.itemgetter``, ``operator.methodcaller``
- ``object.__setattr__``, ``object.__getattribute__``, ``object.__delattr__``, ``.__dict__``, ``.__mro__``, ``.__subclasses__``, ``.__class__``
- Direct dunder method invocations ``obj.__<name>__(...)`` (only ``super().__<name>__(...)`` is permitted)
- Dynamic ``def __getattr__`` definitions
- ``typing.Any``, ``typing_extensions.Any``
- ``typing.cast``, ``typing_extensions.cast``
- ``# type: ignore`` comments
"""

from __future__ import annotations

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

_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})
_FORBIDDEN_TYPING_NAMES: frozenset[str] = frozenset({"Any", "cast"})
_FORBIDDEN_BARE_NAMES: frozenset[str] = frozenset({"Any", "object"})


class _TypingAndReflectionVisitor(ast.NodeVisitor):
  """AST visitor collecting static typing, anti-reflection, and anti-dunder violations."""

  def __init__(self, file_path: Path) -> None:
    """Initialize the visitor.

    Args:
        file_path: Path of the file being checked.
    """
    self.file_path = file_path
    self.violations: list[str] = []

  def _record(self, lineno: int, col_offset: int, message: str) -> None:
    """Record a lint violation.

    Args:
        lineno: 1-based line number.
        col_offset: 0-based column offset.
        message: Human-readable violation description.
    """
    self.violations.append(f"{self.file_path}:{lineno}:{col_offset + 1}: {message}")

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    """Check for forbidden dynamic dunder method definitions like ``__getattr__``.

    Args:
        node: FunctionDef AST node.
    """
    if node.name in _FORBIDDEN_DUNDER_DEFS:
      self._record(
        node.lineno,
        node.col_offset,
        f"Forbidden dynamic method definition 'def {node.name}'",
      )
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    """Check ``from typing import Any, cast``, ``from operator import attrgetter``, and ``from builtins import ...``.

    Args:
        node: ImportFrom AST node.
    """
    module = node.module or ""
    if module in _TYPING_MODULES:
      for alias in node.names:
        if alias.name in _FORBIDDEN_TYPING_NAMES:
          self._record(
            node.lineno,
            node.col_offset,
            f"Forbidden import '{module}.{alias.name}'",
          )
    elif module == "builtins":
      for alias in node.names:
        if alias.name in _FORBIDDEN_BUILTIN_CALLS:
          self._record(
            node.lineno,
            node.col_offset,
            f"Forbidden import 'builtins.{alias.name}'",
          )
    elif module in _OPERATOR_MODULES:
      for alias in node.names:
        if alias.name in _FORBIDDEN_OPERATOR_NAMES:
          self._record(
            node.lineno,
            node.col_offset,
            f"Forbidden import '{module}.{alias.name}'",
          )
    self.generic_visit(node)

  def visit_Attribute(self, node: ast.Attribute) -> None:
    """Check ``typing.Any``, ``typing.cast``, ``operator.attrgetter``, ``builtins.getattr``, and dunder attributes.

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
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden dunder attribute access '.{node.attr}'",
        )
    elif isinstance(node.value, ast.Name):
      if node.value.id in _TYPING_MODULES and node.attr in _FORBIDDEN_TYPING_NAMES:
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden reference '{node.value.id}.{node.attr}'",
        )
      elif node.value.id == "builtins" and node.attr in _FORBIDDEN_BUILTIN_CALLS:
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden builtin reference 'builtins.{node.attr}'",
        )
      elif node.value.id in _OPERATOR_MODULES and node.attr in _FORBIDDEN_OPERATOR_NAMES:
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden operator reflection reference '{node.value.id}.{node.attr}'",
        )
    self.generic_visit(node)

  def visit_Call(self, node: ast.Call) -> None:
    """Check direct calls to forbidden builtins, operator helpers, and direct dunder method invocations.

    Args:
        node: Call AST node.
    """
    if isinstance(node.func, ast.Name):
      if node.func.id in _FORBIDDEN_BUILTIN_CALLS or node.func.id in _FORBIDDEN_OPERATOR_NAMES:
        self._record(
          node.lineno,
          node.col_offset,
          f"Forbidden reflection/cast call '{node.func.id}()'",
        )
    elif isinstance(node.func, ast.Attribute):
      attr = node.func.attr
      if len(attr) > 4 and attr.startswith("__") and attr.endswith("__"):
        is_super = (
          isinstance(node.func.value, ast.Call)
          and isinstance(node.func.value.func, ast.Name)
          and node.func.value.func.id == "super"
        )
        if not is_super:
          self._record(
            node.lineno,
            node.col_offset,
            f"Forbidden direct dunder method call '.{attr}()'",
          )
    self.generic_visit(node)

  def visit_Name(self, node: ast.Name) -> None:
    """Check bare ``Any`` and ``object`` references.

    Args:
        node: Name AST node.
    """
    if node.id in _FORBIDDEN_BARE_NAMES:
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


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
