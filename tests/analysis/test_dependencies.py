"""Test module."""

import libcst as cst
from unittest.mock import patch

from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def test_dependency_scanner():
  """Test element."""
  semantics = SemanticsManager()
  semantics.import_data = {"numpy.core": {}, "pandas": {}}

  scanner = DependencyScanner(semantics, source_fw="torch")

  code = """
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
  tree = cst.parse_module(code)
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


def test_get_root_package_fallback():
  """Test element."""
  semantics = SemanticsManager()
  scanner = DependencyScanner(semantics, source_fw="torch")

  # _get_root_package on non-Name/Attribute
  res = scanner._get_root_package(cst.Integer("1"))
  assert res == ""

  # visit_ImportFrom with no module (e.g. from . import *)
  code = "from . import *"
  tree = cst.parse_module(code)
  tree.visit(scanner)
  # Should not crash, just returns early on node.relative

  # Simulate from ... import without relative, but module is None (invalid CST normally, but test logic)
  class MockImportFrom:
    relative = ()
    module = None

  scanner.visit_ImportFrom(MockImportFrom())  # type: ignore


def test_validate_package_empty():
  """Test element."""
  semantics = SemanticsManager()
  scanner = DependencyScanner(semantics, source_fw="torch")
  scanner._validate_package("")
  assert len(scanner.unknown_imports) == 0


@patch("sys.version_info", (3, 9))
@patch("sys.builtin_module_names", ("sys",))
def test_is_stdlib_py39_fallback():
  """Test element."""
  semantics = SemanticsManager()
  scanner = DependencyScanner(semantics, source_fw="torch")

  # In mocked Python 3.9
  assert scanner._is_stdlib("os") is True  # in common_stdlib
  assert scanner._is_stdlib("sys") is True  # in builtin_module_names
  assert scanner._is_stdlib("cv2") is False
