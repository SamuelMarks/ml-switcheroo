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
  semantics.import_data = {"numpy.core": {}, "pandas": {}}

  scanner: DependencyScanner = DependencyScanner(semantics, source_fw="torch")

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
  scanner: DependencyScanner = DependencyScanner(semantics, source_fw="torch")

  # _get_root_package on non-Name/Attribute
  res: str = scanner._get_root_package(cst.Integer("1"))
  assert res == ""

  # visit_ImportFrom with no module (e.g. from . import *)
  code: str = "from . import *"
  tree: cst.Module = cst.parse_module(code)
  tree.visit(scanner)
  # Should not crash, just returns early on node.relative

  # Simulate from ... import without relative, but module is None (invalid CST normally, but test logic)
  from typing import Optional, Tuple, Union

  class MockImportFrom(cst.ImportFrom):
    """Mock element representing an invalid ImportFrom node with a missing module."""

    def __init__(self) -> None:
      """Initialize the mock ImportFrom element."""
      pass

    @property
    def relative(self) -> Tuple[cst.Dot, ...]:
      """Return an empty tuple to simulate an absolute import."""
      return ()

    @property
    def module(self) -> Optional[Union[cst.Name, cst.Attribute]]:
      """Return None to simulate a missing module node."""
      return None

  scanner.visit_ImportFrom(MockImportFrom())


def test_validate_package_empty() -> None:
  """Test that empty package names are ignored.

  Verifies that calling `_validate_package` with an empty string does not append anything
  to the list of unknown imports.
  """
  semantics: SemanticsManager = SemanticsManager()
  scanner: DependencyScanner = DependencyScanner(semantics, source_fw="torch")
  scanner._validate_package("")
  assert len(scanner.unknown_imports) == 0


@patch("sys.version_info", (3, 9))
@patch("sys.builtin_module_names", ("sys",))
def test_is_stdlib_py39_fallback() -> None:
  """Test the Python 3.9 fallback logic for standard library detection.

  Prior to Python 3.10, `sys.stdlib_module_names` does not exist. This test mocks
  Python 3.9 and verifies that `_is_stdlib` correctly uses a hardcoded fallback list
  and `sys.builtin_module_names` to identify stdlib modules like `os` and `sys`.
  """
  semantics: SemanticsManager = SemanticsManager()
  scanner: DependencyScanner = DependencyScanner(semantics, source_fw="torch")

  # In mocked Python 3.9
  assert scanner._is_stdlib("os") is True  # in common_stdlib
  assert scanner._is_stdlib("sys") is True  # in builtin_module_names
  assert scanner._is_stdlib("cv2") is False


# --- Merged from test_dependencies_extra.py ---


def test_is_stdlib_py310(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test standard library detection using sys.stdlib_module_names on Python 3.10+.

  This test mocks Python 3.10 and ensures that `_is_stdlib` utilizes the modern, built-in
  set `sys.stdlib_module_names` to definitively check if a module is in the standard library.
  """
  monkeypatch.setattr(sys, "version_info", (3, 10))
  # We must patch sys.stdlib_module_names
  monkeypatch.setattr(sys, "stdlib_module_names", {"os", "sys"}, raising=False)

  scanner: DependencyScanner = DependencyScanner(MagicMock(), "torch")
  assert scanner._is_stdlib("os") is True
  assert scanner._is_stdlib("torch") is False
