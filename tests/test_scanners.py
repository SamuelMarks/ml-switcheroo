"""Tests for usage scanners."""

import libcst as cst

from ml_switcheroo.core.scanners import GlobalUsageScanner


def test_global_usage_scanner_attribute() -> None:
  """Docstring."""
  code: str = "import torch.nn as nn\nself.conv = keras.layers.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: GlobalUsageScanner = GlobalUsageScanner()
  module.visit(scanner)
  assert "nn" not in scanner.used_names
  assert "keras" in scanner.used_names


def test_global_usage_scanner_attribute_used() -> None:
  """Docstring."""
  code: str = "import torch.nn as nn\nself.conv = nn.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: GlobalUsageScanner = GlobalUsageScanner()
  module.visit(scanner)
  assert "nn" in scanner.used_names


def test_global_usage_scanner_nested_attribute_used() -> None:
  """Docstring."""
  code: str = "import torch\nself.conv = torch.nn.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: GlobalUsageScanner = GlobalUsageScanner()
  module.visit(scanner)
  assert "torch" in scanner.used_names
