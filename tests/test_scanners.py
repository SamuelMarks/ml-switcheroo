"""Tests for usage scanners."""

import libcst as cst

from ml_switcheroo.core.scanners import UsageScanner


def test_usage_scanner_attribute() -> None:
  """Docstring."""
  code: str = "import torch.nn as nn\nself.conv = keras.layers.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: UsageScanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_attribute_used() -> None:
  """Docstring."""
  code: str = "import torch.nn as nn\nself.conv = nn.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: UsageScanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()


def test_usage_scanner_nested_attribute_used() -> None:
  """Docstring."""
  code: str = "import torch\nself.conv = torch.nn.Conv2D(32, 3)"
  module: cst.Module = cst.parse_module(code)
  scanner: UsageScanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
