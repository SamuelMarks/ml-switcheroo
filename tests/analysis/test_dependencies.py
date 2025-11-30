"""
Tests for Import Dependency Validator (Feature 055).

Verifies that:
1. Standard library imports (e.g., `os`, `json`) are ignored.
2. Source framework imports (e.g., `torch`) are ignored (handled by core engine).
3. Mapped imports (present in SemanticsManager) are ignored.
4. Unmapped 3rd-party imports (e.g., `pandas`, `cv2`) are FLAGGED.
"""

import pytest
import libcst as cst

from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock semantics with predefined import mappings."""

  def __init__(self):
    self.import_data = {
      # Known mappings
      "numpy": {},  # Basic root
      "PIL.Image": {},  # Submodule, implies PIL is known
      "optax": {},
    }


@pytest.fixture
def scanner():
  semantics = MockSemantics()
  # Source framework is "torch", targeting something else
  return DependencyScanner(semantics, source_fw="torch")


def scan_code(scanner, code):
  """Heper to parse code and run scanner."""
  tree = cst.parse_module(code)
  tree.visit(scanner)
  return scanner.unknown_imports


def test_ignore_stdlib(scanner):
  """
  Scenario: User imports 'os', 'sys', 'typing'.
  Expectation: No warnings (all detected as stdlib).
  """
  code = """
import os
import sys
from typing import List
from datetime import datetime
"""
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_ignore_source_framework(scanner):
  """
  Scenario: User imports 'torch' or 'torch.nn'.
  Expectation: No warnings (handled by ImportFixer).
  """
  code = """
import torch
import torch.nn as nn
from torch import optim
"""
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_ignore_mapped_dependencies(scanner):
  """
  Scenario: User imports 'numpy' and 'PIL'.
  Expectation: No warnings (present in MockSemantics).
  """
  code = """
import numpy as np
import PIL
from PIL import Image
"""
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_flag_unmapped_third_party(scanner):
  """
  Scenario: User imports 'pandas' and 'cv2'.
  Expectation: Flagged as unknown dependencies.
  """
  code = """
import pandas as pd
import cv2
"""
  unknowns = scan_code(scanner, code)
  assert "pandas" in unknowns
  assert "cv2" in unknowns
  assert len(unknowns) == 2


def test_flag_deep_imports(scanner):
  """
  Scenario: `from sklearn.metrics import f1_score`.
  Expectation: Flag 'sklearn' as unknown root.
  """
  code = "from sklearn.metrics import f1_score"
  unknowns = scan_code(scanner, code)
  assert "sklearn" in unknowns


def test_ignore_relative_imports(scanner):
  """
  Scenario: `from . import utils` or `from ..models import net`.
  Expectation: Ignored (internal project structure).
  """
  # LibCST ImportFrom.module is None for 'from . import x',
  # or starts with dots for 'from .x import y'.
  # Our _get_root_package handles names.

  # AST: from . import x -> module=None
  code1 = "from . import x"
  unknowns1 = scan_code(scanner, code1)
  assert len(unknowns1) == 0

  # AST: from .sub import y -> module is an Attribute/Name but LibCST structures relative differently?
  # Actually, LibCST `ImportFrom` has `relative: Sequence[Dot]`.
  # `module` can be None.
  # scan_code relies on `if node.module:`.

  code2 = "from .sub import y"
  unknowns2 = scan_code(scanner, code2)
  assert len(unknowns2) == 0
