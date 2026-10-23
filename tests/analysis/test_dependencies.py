"""Test module for the semantic API DependencyScanner.

This module contains unit tests verifying the correctness of `DependencyScanner` in identifying
external library dependencies from CST trees. It tests standard library filtering, semantic
root resolution (ignoring known ML frameworks), and detection of completely unknown modules.
"""

import sys
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def test_dependency_scanner() -> None:
  """Test standard dependency extraction and categorizations.

  This test provides a block of imports and verifies that the DependencyScanner
  correctly categorizes them. It checks that `cv2` and `unknown_lib` are identified
  as unknown imports, while standard libraries (`os`, `sys`), the source framework
  (`torch`), relative imports, and known semantic targets (`numpy`, `pandas`) are safely ignored.
  """
  semantics: SemanticsManager = SemanticsManager()
  setattr(semantics, "import_data", {"numpy.core": {}, "pandas": {}})

  scanner: DependencyScanner = DependencyScanner(semantics, source_framework="torch")

  code: str = """
import os
import sys
import torch
import torch.nn
from torch import nn
from . import relative
from cv2 import imread
import unknown_lib.submodule
import numpy as np
import pandas as pd
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(scanner)

  # Check known roots are cached correctly
  assert "numpy" in scanner._known_semantic_roots
  assert "pandas" in scanner._known_semantic_roots

  # Check unknown imports
  assert "cv2" in scanner.unknown_imports
  assert "unknown_lib" in scanner.unknown_imports

  # Check ignored imports
  assert "os" not in scanner.unknown_imports  # stdlib
  assert "sys" not in scanner.unknown_imports  # stdlib
  assert "torch" not in scanner.unknown_imports  # source fw
  assert "relative" not in scanner.unknown_imports  # not parsed due to node.relative
  assert "numpy" not in scanner.unknown_imports  # in semantics
  assert "pandas" not in scanner.unknown_imports  # in semantics


def test_get_root_package_fallback() -> None:
  """Test edge cases and fallback logic for extracting root packages from AST nodes.

  This test ensures that `_get_root_package` returns an empty string when passed an
  invalid node (like `cst.Integer`), and that the scanner handles malformed or complex
  `ImportFrom` statements gracefully without crashing.
  """
  semantics: SemanticsManager = SemanticsManager()
  scanner: DependencyScanner = DependencyScanner(semantics, source_framework="torch")

  # _get_root_package on non-Name/Attribute
  res: str = scanner._get_root_package(cst.Integer("1"))
  assert res == ""

  # visit_ImportFrom with no module (e.g. from . import *)
  code: str = "from . import *"
  tree: cst.Module = cst.parse_module(code)
  tree.visit(scanner)

  class MockImportFrom(cst.ImportFrom):
    """Mock element representing an invalid ImportFrom node with a missing module."""

    def __init__(self) -> None:
      """Initialize the mock ImportFrom element."""
      pass

    @property
    def relative(self) -> tuple[cst.Dot, ...]:
      """Return an empty tuple to simulate an absolute import.

      Returns:
          The `relative` property value (tuple[cst.Dot, ...]).
      """
      return ()

    @property
    def module(self) -> cst.Name | cst.Attribute | None:
      """Return None to simulate a missing module node.

      Returns:
          The `module` property value (cst.Name | cst.Attribute | None).
      """
      return None

  scanner.visit_ImportFrom(MockImportFrom())


def test_validate_package_empty() -> None:
  """Test that empty package names are ignored.

  Verifies that calling `_validate_package` with an empty string does not append anything
  to the list of unknown imports.
  """
  semantics: SemanticsManager = SemanticsManager()
  scanner: DependencyScanner = DependencyScanner(semantics, source_framework="torch")
  scanner._validate_package("")
  assert len(scanner.unknown_imports) == 0


@patch("sys.stdlib_module_names", frozenset({"os"}))
@patch("sys.builtin_module_names", ("sys",))
def test_is_stdlib_builtin_and_stdlib() -> None:
  """Test standard library and builtin module detection on Python 3.10+.

  Verifies that `_is_stdlib` checks both `sys.stdlib_module_names` and
  `sys.builtin_module_names` to identify stdlib modules like `os` and `sys`.
  """
  semantics: SemanticsManager = SemanticsManager()
  scanner: DependencyScanner = DependencyScanner(semantics, source_framework="torch")

  assert scanner._is_stdlib("os") is True  # in stdlib_module_names
  assert scanner._is_stdlib("sys") is True  # in builtin_module_names
  assert scanner._is_stdlib("cv2") is False


def test_is_stdlib_py310(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test standard library detection using sys.stdlib_module_names on Python 3.10+.

  This test mocks Python 3.10 and ensures that `_is_stdlib` utilizes the modern, built-in
  set `sys.stdlib_module_names` to definitively check if a module is in the standard library.

  Args:
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  monkeypatch.setattr(sys, "version_info", (3, 10))
  # We must patch sys.stdlib_module_names
  monkeypatch.setattr(sys, "stdlib_module_names", {"os", "sys"}, raising=False)

  scanner: DependencyScanner = DependencyScanner(MagicMock(), "torch")
  assert scanner._is_stdlib("os") is True
  assert scanner._is_stdlib("torch") is False
