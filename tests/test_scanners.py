"""Tests for usage scanners."""

import libcst as cst
from ml_switcheroo.core.scanners import UsageScanner


def test_usage_scanner_attribute():
  """Test usage scanner ignores unrelated attributes."""
  code = "import torch.nn as nn\nself.conv = keras.layers.Conv2D(32, 3)"
  module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_attribute_used():
  """Test usage scanner finds used attributes."""
  code = "import torch.nn as nn\nself.conv = nn.Conv2D(32, 3)"
  module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()


def test_usage_scanner_nested_attribute_used():
  """Test usage scanner finds nested used attributes."""
  code = "import torch\nself.conv = torch.nn.Conv2D(32, 3)"
  module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
