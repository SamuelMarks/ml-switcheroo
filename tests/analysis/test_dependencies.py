"""Test suite for the Dependencies module."""

import pytest
import sys
from unittest.mock import patch
import libcst as cst
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.import_data = {"numpy": {}, "PIL.Image": {}, "optax": {}}


@pytest.fixture
def scanner():
  """Provides a mock scanner for testing."""
  semantics = MockSemantics()
  return DependencyScanner(semantics, source_fw="torch")


def scan_code(scanner, code):
  """Scans code."""
  tree = cst.parse_module(code)
  tree.visit(scanner)
  return scanner.unknown_imports


def test_ignore_stdlib(scanner):
  """Verifies the behavior of ignore stdlib."""
  code = "\nimport os\nimport sys\nfrom typing import Union, List\nfrom datetime import datetime\n"
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_ignore_source_framework(scanner):
  """Verifies the behavior of ignore source framework."""
  code = "\nimport torch\nimport torch.nn as nn\nfrom torch import optim\n"
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_ignore_mapped_dependencies(scanner):
  """Verifies the behavior of ignore mapped dependencies."""
  code = "\nimport numpy as np\nimport PIL\nfrom PIL import Image\n"
  unknowns = scan_code(scanner, code)
  assert len(unknowns) == 0


def test_flag_unmapped_third_party(scanner):
  """Verifies the behavior of flag unmapped third party."""
  code = "\nimport pandas as pd\nimport cv2\n"
  unknowns = scan_code(scanner, code)
  assert "pandas" in unknowns
  assert "cv2" in unknowns
  assert len(unknowns) == 2


def test_flag_deep_imports(scanner):
  """Verifies the behavior of flag deep imports."""
  code = "from sklearn.metrics import f1_score"
  unknowns = scan_code(scanner, code)
  assert "sklearn" in unknowns


def test_ignore_relative_imports(scanner):
  """Verifies the behavior of ignore relative imports."""
  code1 = "from . import x"
  unknowns1 = scan_code(scanner, code1)
  assert len(unknowns1) == 0
  code2 = "from .sub import y"
  unknowns2 = scan_code(scanner, code2)
  assert len(unknowns2) == 0


def test_get_root_package_non_name(scanner):
  """Gets root package non name."""
  res = scanner._get_root_package(cst.Integer("1"))
  assert res == ""


def test_validate_package_empty(scanner):
  """Validates package empty."""
  scanner._validate_package("")
  assert len(scanner.unknown_imports) == 0


def test_is_stdlib_fallback(scanner):
  """Checks if is stdlib fallback."""
  with patch.object(sys, "version_info", (3, 9)):
    assert scanner._is_stdlib("os") is True
    assert scanner._is_stdlib("unknown_lib") is False
    with patch.object(sys, "builtin_module_names", ["fake_builtin"]):
      assert scanner._is_stdlib("fake_builtin") is True


def test_no_semantics():
  """Verifies the behavior of no semantics."""
  scanner = DependencyScanner(None, source_fw="torch")
  assert len(scanner._known_semantic_roots) == 0
