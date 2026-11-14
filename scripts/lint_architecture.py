"""AST-based Architectural Boundary Linter for ``ml-switcheroo``.

Replaces fragile regex heuristics with full Python AST static analysis enforcing:

1. **Zero Hardcoded Framework/Dialect Literals in Core**: Bans string literals matching
   any registered framework or dialect (``torch``, ``jax``, ``flax``, ``flax_nnx``,
   ``mlx``, ``keras``, ``tensorflow``, ``numpy``, ``paxml``, ``nvidia_sass``, ``rdna``,
   ``stablehlo``, ``mlir``, ``ir``, ``tikz``, ``html``, ``latex_dsl``, ``cpp``, ``wasm``)
   inside ``ml_switcheroo/core/`` (outside explicit dialect packages and registry/strategy
   routing tables) and framework dictionary keys in ``ml_switcheroo/plugins/``.
2. **Zero Hardcoded Framework Routing Comparisons**: Bans ``framework/source/target == '<fw>'``
   and ``in {'<fw>', ...}`` dispatch checks outside explicit registry/strategy tables.
3. **Zero Cross-Layer Imports into Core**: Bans importing ``ml_switcheroo.cli`` or
   ``ml_switcheroo.testing`` inside ``ml_switcheroo/core/``.
4. **Zero Unauthorized ``sys.modules`` / ``sys.path`` Mutations**: Bans mutating
   ``sys.modules`` or ``sys.path`` outside designated bootstrap/isolation modules.
5. **Zero Reflection Dunder Type-Checker Evasions**: Bans ``getattr({"v": ...}, ...)`` and
   ``getattr(..., "__getitem__" | "__setitem__" | "__call__")``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Sequence

REGISTERED_FRAMEWORKS_AND_DIALECTS: frozenset[str] = frozenset(
  {
    "torch",
    "jax",
    "flax",
    "flax_nnx",
    "mlx",
    "keras",
    "tensorflow",
    "numpy",
    "paxml",
    "nvidia_sass",
    "rdna",
    "stablehlo",
    "mlir",
    "ir",
    "tikz",
    "html",
    "latex_dsl",
    "cpp",
    "wasm",
  }
)

ROUTING_VAR_NAMES: frozenset[str] = frozenset(
  {
    "framework",
    "fw",
    "fw_key",
    "target",
    "source",
    "target_framework",
    "source_framework",
    "target_fw",
    "source_fw",
  }
)

CORE_REGISTRY_ALLOWLIST_SUFFIXES: tuple[str, ...] = (
  "ml_switcheroo/core/compiler/registry.py",
  "ml_switcheroo/core/strategies.py",
  "ml_switcheroo/core/engine.py",
  "ml_switcheroo/core/ingestion.py",
  "ml_switcheroo/core/ir_bridge.py",
)

CORE_DIALECT_SUBPATH_PREFIXES: tuple[str, ...] = (
  "ml_switcheroo/core/ir_bridge/",
  "ml_switcheroo/core/mlir/",
  "ml_switcheroo/core/wasm/",
  "ml_switcheroo/core/html/",
  "ml_switcheroo/core/latex/",
  "ml_switcheroo/core/tikz/",
  "ml_switcheroo/core/markup/",
  "ml_switcheroo/core/compiler/backends/nvidia_sass/",
  "ml_switcheroo/core/compiler/backends/rdna/",
  "ml_switcheroo/core/compiler/backends/stablehlo.py",
  "ml_switcheroo/core/compiler/backends/compiler_bridge.py",
  "ml_switcheroo/core/compiler/backends/cpp/",
  "ml_switcheroo/core/compiler/frontends/nvidia_sass/",
  "ml_switcheroo/core/compiler/frontends/rdna/",
)

ROUTING_ALLOWLIST_SUFFIXES: tuple[str, ...] = (
  *CORE_REGISTRY_ALLOWLIST_SUFFIXES,
  *CORE_DIALECT_SUBPATH_PREFIXES,
  "ml_switcheroo/config.py",
  "ml_switcheroo/generated_tests/runtime_builder.py",
  "ml_switcheroo/generated_tests/inputs.py",
  "ml_switcheroo/testing/harness_generator.py",
  "scripts/lint_architecture.py",
)

SYS_MUTATION_ALLOWLIST_SUFFIXES: tuple[str, ...] = (
  "ml_switcheroo/core/bootstrap.py",
  "ml_switcheroo/core/hooks_registry.py",
)

FORBIDDEN_CORE_IMPORT_PREFIXES: tuple[str, ...] = (
  "ml_switcheroo.cli",
  "ml_switcheroo.testing",
)

MUTATING_CONTAINER_METHODS: frozenset[str] = frozenset(
  {
    "append",
    "insert",
    "extend",
    "pop",
    "clear",
    "update",
    "setdefault",
    "remove",
  }
)

EVASION_DUNDERS: frozenset[str] = frozenset({"__getitem__", "__setitem__", "__call__"})


def _normalize_posix(file_path: Path) -> str:
  """Normalize ``file_path`` to a forward-slash POSIX path string.

  Args:
      file_path: Candidate file path.

  Returns:
      str: Forward-slash normalized path string.
  """
  return file_path.as_posix()


def _matches_any_segment(posix_path: str, patterns: Sequence[str]) -> bool:
  """Check whether ``posix_path`` ends with or contains any subpath in ``patterns``.

  Args:
      posix_path: Normalized POSIX path string.
      patterns: Suffix or directory prefix patterns.

  Returns:
      bool: ``True`` if any pattern matches ``posix_path``.
  """
  return any(pat in posix_path for pat in patterns)


def _docstring_expr_ids(tree: ast.AST) -> set[int]:
  """Collect ``id(node)`` for all module, class, and function docstring ``Constant`` nodes.

  Args:
      tree: Parsed Python AST root.

  Returns:
      set[int]: Object IDs of docstring string constant AST nodes.
  """
  doc_ids: set[int] = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
      first = node.body[0]
      if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        doc_ids.add(id(first.value))
  return doc_ids


def _extract_var_leaf(expr: ast.expr) -> str | None:
  """Extract the leaf variable/attribute name from ``Name``, ``Attribute``, or ``str(...)`` call.

  Args:
      expr: Candidate comparison operand AST node.

  Returns:
      str | None: Leaf identifier if recognized, otherwise ``None``.
  """
  if isinstance(expr, ast.Name):
    return expr.id
  if isinstance(expr, ast.Attribute):
    return expr.attr
  if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "str" and len(expr.args) == 1:
    return _extract_var_leaf(expr.args[0])
  return None


def _contains_framework_literal(expr: ast.expr) -> str | None:
  """Check whether ``expr`` is a framework literal or a literal collection containing one.

  Args:
      expr: Comparison right-hand or left-hand AST expression.

  Returns:
      str | None: Matched framework/dialect string if present, else ``None``.
  """
  if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
    if expr.value in REGISTERED_FRAMEWORKS_AND_DIALECTS:
      return expr.value
  elif isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
    for elt in expr.elts:
      if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and elt.value in REGISTERED_FRAMEWORKS_AND_DIALECTS:
        return elt.value
  return None


class _ArchitectureVisitor(ast.NodeVisitor):
  """AST visitor enforcing architectural boundaries, layering, and isolation invariants.

  Attributes:
      file_path: Inspected file path (`Path`).
      posix_path: Normalized POSIX file path (`str`).
      violations: Collected human-readable violation strings (`list[str]`).
  """

  def __init__(self, file_path: Path, docstring_ids: set[int]) -> None:
    """Initialize ``_ArchitectureVisitor``.

    Args:
        file_path: Path to the Python source file being linted.
        docstring_ids: Set of ``id(...)`` values for docstring ``Constant`` nodes.
    """
    self.file_path: Path = file_path
    self.posix_path: str = _normalize_posix(file_path)
    self.violations: list[str] = []
    self._docstring_ids: set[int] = docstring_ids
    self._dict_key_ids: set[int] = set()
    self._sys_aliases: set[str] = {"sys"}
    self._sys_target_aliases: dict[str, str] = {}
    self._is_core: bool = "ml_switcheroo/core/" in self.posix_path
    self._is_plugin: bool = "ml_switcheroo/plugins/" in self.posix_path
    self._is_script: bool = self.posix_path.startswith("scripts/") or "/scripts/" in self.posix_path
    self._allow_core_literals: bool = _matches_any_segment(
      self.posix_path,
      (*CORE_REGISTRY_ALLOWLIST_SUFFIXES, *CORE_DIALECT_SUBPATH_PREFIXES),
    )
    self._allow_routing: bool = self._is_script or _matches_any_segment(
      self.posix_path,
      ROUTING_ALLOWLIST_SUFFIXES,
    )
    self._allow_sys_mutation: bool = self._is_script or _matches_any_segment(
      self.posix_path,
      SYS_MUTATION_ALLOWLIST_SUFFIXES,
    )

  def _add(self, node: ast.AST, prefix: str, detail: str) -> None:
    """Record a formatted violation message on ``self.violations``.

    Args:
        node: AST node carrying line number metadata.
        prefix: Canonical violation category header.
        detail: Explanation of the violation.
    """
    lineno = node.lineno if isinstance(node, (ast.stmt, ast.expr)) else 1
    self.violations.append(f"{prefix} in {self.file_path}:{lineno}: {detail}")

  def _is_sys_state_expr(self, expr: ast.expr) -> str | None:
    """Return ``'sys.modules'`` or ``'sys.path'`` if ``expr`` references either global structure.

    Args:
        expr: Candidate target or receiver expression.

    Returns:
        str | None: ``'sys.modules'`` or ``'sys.path'`` when matched, otherwise ``None``.
    """
    if isinstance(expr, ast.Name) and expr.id in self._sys_target_aliases:
      return self._sys_target_aliases[expr.id]
    if (
      isinstance(expr, ast.Attribute)
      and isinstance(expr.value, ast.Name)
      and expr.value.id in self._sys_aliases
      and expr.attr in ("modules", "path")
    ):
      return f"sys.{expr.attr}"
    return None

  def visit_Import(self, node: ast.Import) -> None:
    """Inspect ``import ...`` statements for cross-layer imports and ``sys`` aliases.

    Args:
        node: ``ast.Import`` node.
    """
    for alias in node.names:
      if alias.name == "sys":
        self._sys_aliases.add(alias.asname or "sys")
      if self._is_core and alias.name.startswith(FORBIDDEN_CORE_IMPORT_PREFIXES):
        self._add(
          node,
          "Error: Cross-layer import found",
          f"core module must not import '{alias.name}'",
        )
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    """Inspect ``from ... import ...`` statements for cross-layer imports and ``sys`` aliases.

    Args:
        node: ``ast.ImportFrom`` node.
    """
    mod = node.module or ""
    if mod == "sys":
      for alias in node.names:
        if alias.name in ("modules", "path"):
          self._sys_target_aliases[alias.asname or alias.name] = f"sys.{alias.name}"
    if self._is_core and mod.startswith(FORBIDDEN_CORE_IMPORT_PREFIXES):
      self._add(
        node,
        "Error: Cross-layer import found",
        f"core module must not import from '{mod}'",
      )
    self.generic_visit(node)

  def visit_Dict(self, node: ast.Dict) -> None:
    """Inspect dictionary keys for hardcoded framework names inside ``core/`` and ``plugins/``.

    Args:
        node: ``ast.Dict`` node.
    """
    for key in node.keys:
      if isinstance(key, ast.Constant) and isinstance(key.value, str):
        self._dict_key_ids.add(id(key))
        if key.value in REGISTERED_FRAMEWORKS_AND_DIALECTS and (
          (self._is_core and not self._allow_core_literals) or self._is_plugin
        ):
          self._add(
            key,
            "Error: Hardcoded framework dictionary key found",
            f"key '{key.value}' must live in StructuralTraits, PluginTraits, or ODL",
          )
    self.generic_visit(node)

  def visit_Constant(self, node: ast.Constant) -> None:
    """Inspect string literals inside ``ml_switcheroo/core/`` for leaked framework names.

    Args:
        node: ``ast.Constant`` node.
    """
    if (
      self._is_core
      and not self._allow_core_literals
      and id(node) not in self._docstring_ids
      and id(node) not in self._dict_key_ids
      and isinstance(node.value, str)
      and node.value in REGISTERED_FRAMEWORKS_AND_DIALECTS
    ):
      self._add(
        node,
        "Error: Hardcoded framework literal found",
        f"literal '{node.value}' is forbidden inside ml_switcheroo.core",
      )
    self.generic_visit(node)

  def visit_Compare(self, node: ast.Compare) -> None:
    """Inspect ``==``, ``!=``, ``in``, and ``not in`` comparisons for hardcoded framework routing.

    Args:
        node: ``ast.Compare`` node.
    """
    if not self._allow_routing:
      operands = [node.left, *node.comparators]
      for idx, op in enumerate(node.ops):
        if isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
          lhs = operands[idx]
          rhs = operands[idx + 1]
          lhs_var = _extract_var_leaf(lhs)
          rhs_fw = _contains_framework_literal(rhs)
          rhs_var = _extract_var_leaf(rhs)
          lhs_fw = _contains_framework_literal(lhs)
          matched_fw = (
            rhs_fw
            if (lhs_var in ROUTING_VAR_NAMES and rhs_fw)
            else (lhs_fw if (rhs_var in ROUTING_VAR_NAMES and lhs_fw) else None)
          )
          if matched_fw is not None:
            self._add(
              node,
              "Error: Hardcoded framework routing found",
              f"comparison against '{matched_fw}' must delegate to traits or ODL variants",
            )
    self.generic_visit(node)

  def _check_mutation_target(self, target: ast.expr, stmt: ast.stmt) -> None:
    """Check an assignment or deletion target for unauthorized ``sys.modules`` / ``sys.path`` mutation.

    Args:
        target: Assignment/deletion/augassign target AST node.
        stmt: Enclosing statement node.
    """
    if self._allow_sys_mutation:
      return
    direct = self._is_sys_state_expr(target)
    if direct is not None:
      self._add(stmt, "Error: Unauthorized sys mutation found", f"direct mutation of '{direct}' is forbidden")
      return
    if isinstance(target, ast.Subscript):
      sub_state = self._is_sys_state_expr(target.value)
      if sub_state is not None:
        self._add(stmt, "Error: Unauthorized sys mutation found", f"item mutation on '{sub_state}' is forbidden")

  def visit_Assign(self, node: ast.Assign) -> None:
    """Check assignment targets for ``sys.modules`` or ``sys.path`` mutation.

    Args:
        node: ``ast.Assign`` node.
    """
    for tgt in node.targets:
      self._check_mutation_target(tgt, node)
    self.generic_visit(node)

  def visit_AugAssign(self, node: ast.AugAssign) -> None:
    """Check augmented assignment targets (e.g. ``sys.path += [...]``) for ``sys`` mutation.

    Args:
        node: ``ast.AugAssign`` node.
    """
    self._check_mutation_target(node.target, node)
    self.generic_visit(node)

  def visit_Delete(self, node: ast.Delete) -> None:
    """Check ``del`` targets (e.g. ``del sys.modules[...]``) for ``sys`` mutation.

    Args:
        node: ``ast.Delete`` node.
    """
    for tgt in node.targets:
      self._check_mutation_target(tgt, node)
    self.generic_visit(node)

  def visit_Call(self, node: ast.Call) -> None:
    """Check method calls for ``sys`` mutations and reflection dunder type-checker evasions.

    Args:
        node: ``ast.Call`` node.
    """
    if not self._allow_sys_mutation and isinstance(node.func, ast.Attribute):
      state = self._is_sys_state_expr(node.func.value)
      if state is not None and node.func.attr in MUTATING_CONTAINER_METHODS:
        self._add(
          node,
          "Error: Unauthorized sys mutation found",
          f"'{state}.{node.func.attr}(...)' is forbidden outside bootstrap isolation modules",
        )

    is_getattr_call = (isinstance(node.func, ast.Name) and node.func.id == "getattr") or (
      isinstance(node.func, ast.Attribute) and node.func.attr == "getattr"
    )
    if is_getattr_call and len(node.args) >= 2:
      arg0, arg1 = node.args[0], node.args[1]
      is_dict_box = (
        isinstance(arg0, ast.Dict)
        and len(arg0.keys) == 1
        and isinstance(arg0.keys[0], ast.Constant)
        and arg0.keys[0].value == "v"
      )
      is_evasion_dunder = isinstance(arg1, ast.Constant) and arg1.value in EVASION_DUNDERS
      if is_dict_box or is_evasion_dunder:
        self._add(
          node,
          "Error: Reflection type-checker evasion found",
          "use explicit Protocol or isinstance narrowing instead of dunder getattr evasion",
        )

    self.generic_visit(node)


def check_source(source: str, file_path: Path) -> list[str]:
  """Run all architectural boundary checks on ``source``.

  Args:
      source: Python source text to analyze.
      file_path: Logical file path used for layer classification and diagnostics.

  Returns:
      list[str]: Collected violation strings (empty when compliant).
  """
  tree = ast.parse(source, filename=str(file_path))
  doc_ids = _docstring_expr_ids(tree)
  visitor = _ArchitectureVisitor(file_path, doc_ids)
  visitor.visit(tree)
  return visitor.violations


def iter_python_files(paths: Sequence[str]) -> list[Path]:
  """Expand CLI file and directory arguments into sorted ``*.py`` paths.

  Args:
      paths: Sequence of file or directory path strings.

  Returns:
      list[Path]: Sorted Python file paths (non-Python files are ignored).
  """
  collected: list[Path] = []
  for raw in paths:
    path = Path(raw)
    if path.is_file():
      if path.suffix == ".py":
        collected.append(path)
    elif path.is_dir():
      collected.extend(sorted(path.rglob("*.py")))
  return collected


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for architectural boundary enforcement.

  Args:
      argv: Optional command-line path list (defaults to ``sys.argv[1:]``).

  Returns:
      int: ``0`` if zero violations were found, ``1`` otherwise.
  """
  raw_args = sys.argv[1:] if argv is None else list(argv)
  if not raw_args:
    return 0

  all_violations: list[str] = []
  for py_file in iter_python_files(raw_args):
    source = py_file.read_text(encoding="utf-8")
    all_violations.extend(check_source(source, py_file))

  if all_violations:
    for violation in all_violations:
      print(violation)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
