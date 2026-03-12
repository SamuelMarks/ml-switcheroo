"""
Static Dependency Analysis for Transpilation Safety.

This module provides the ``DependencyScanner``, a LibCST visitor that identifies
third-party dependencies imported in the source code.

It serves as a safety check during transpilation:

1.  Identifies imports (e.g., ``import pandas``, ``from sklearn import metrics``).
2.  Filters out **Standard Library** modules (e.g., ``os``, ``sys``, ``json``).
3.  Filters out the **Source Framework** (e.g., ``torch``), as these are handled
    by the core ``ImportFixer`` and ``PivotRewriter``.
4.  Validates the remaining imports against the **Semantics Manager**.

If a third-party import (e.g., ``cv2``) is found but not mapped in the
semantics (Import Map), it warns the user that the target environment
might lack the equivalent library or that the mapping logic is missing.
"""

import sys
import libcst as cst
from typing import Set

from ml_switcheroo.semantics.manager import SemanticsManager


class DependencyScanner(cst.CSTVisitor):
  """
  Scans for 3rd-party imports not covered by the Semantic Knowledge Base.
  """

  def __init__(self, semantics: SemanticsManager, source_fw: str):
    """
    Initializes the scanner.

    Args:
        semantics: The semantics manager containing valid import maps.
        source_fw: The root name of the source framework to ignore (transpilation target).
    """
    self.semantics = semantics
    self.source_fw = source_fw
    # Accumulated list of unmapped 3rd-party root packages.
    self.unknown_imports: Set[str] = set()

    # Build local cache of known mapped roots to avoid repeated deep lookups.
    # Logic: Semantics import_data keys are "source.module". We want the roots.
    self._known_semantic_roots = set()
    if semantics and hasattr(semantics, "import_data"):
      for key in semantics.import_data.keys():
        root = key.split(".")[0]
        self._known_semantic_roots.add(root)

  def visit_Import(self, node: cst.Import) -> None:
    """
    Visits ``import x``, ``import x.y``.
    Checks the root package name.

    Args:
        node: The import statement node.
    """
    for alias in node.names:
      root_pkg = self._get_root_package(alias.name)
      self._validate_package(root_pkg)

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Visits ``from x import y``.
    Checks the module root package name.

    Args:
        node: The import-from statement node.
    """
    # Ignore relative imports (starting with dots like `from . import utils`)
    if node.relative:
      return

    if node.module:
      root_pkg = self._get_root_package(node.module)
      self._validate_package(root_pkg)

  def _get_root_package(self, node: cst.BaseExpression) -> str:
    """
    Extracts the root package string from a CST node.
    e.g., ``Attribute(Name(os), Name(path))`` -> "os".

    Args:
        node: The node representing the module name.

    Returns:
        The root identifier string (e.g. 'os') or empty string if not found.
    """
    # Unwrap Attribute chains to find the leftmost Name
    curr = node
    while isinstance(curr, cst.Attribute):
      curr = curr.value

    if isinstance(curr, cst.Name):
      return curr.value

    return ""

  def _validate_package(self, pkg_name: str) -> None:
    """
    Filters and checks the package name.
    If it's external and unmapped, adds to unknown_imports.

    Args:
        pkg_name: The root package identifier.
    """
    if not pkg_name:
      return

    # 1. Ignore Source Framework (Handled by Rewriter)
    if pkg_name == self.source_fw:
      return

    # 2. Ignore Standard Library
    if self._is_stdlib(pkg_name):
      return

    # 3. Check Semantics
    if pkg_name in self._known_semantic_roots:
      return

    # 4. Flag as Unknown
    self.unknown_imports.add(pkg_name)

  def _is_stdlib(self, name: str) -> bool:
    """
    Determines if a package is part of the Python Standard Library.
    Uses ``sys.stdlib_module_names`` on Python 3.10+, falls back to known list.

    Args:
        name: The package name to check.

    Returns:
        True if the package is in the standard library.
    """
    # Python 3.10+
    if sys.version_info >= (3, 10):
      return name in sys.stdlib_module_names

    # Fallback for Python 3.9 (common subset)
    # This list serves as a heuristic for older envs supported by the classifier.
    common_stdlib = frozenset(
      (
        "os",
        "sys",
        "re",
        "math",
        "json",
        "time",
        "datetime",
        "random",
        "typing",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "argparse",
        "subprocess",
        "logging",
        "shutil",
        "glob",
        "unittest",
        "copy",
        "pickle",
        "warnings",
        "abc",
        "contextlib",
        "enum",
        "io",
        "threading",
        "multiprocessing",
        "queue",
        "socket",
        "urllib",
        "http",
        "email",
        "html",
        "xml",
        "csv",
        "hashlib",
        "base64",
        "uuid",
        "stat",
        "tempfile",
        "traceback",
        "inspect",
        "types",
        "ast",
        "dis",
        "gc",
        "platform",
        "shlex",
        "signal",
        "site",
        "sysconfig",
        "weakref",
        "zipfile",
        "zlib",
      )
    )

    # Check builtins (available in sys even in old python)
    if name in sys.builtin_module_names:
      return True

    return name in common_stdlib
