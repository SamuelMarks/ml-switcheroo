"""Tests for AST Scanners."""

import libcst as cst

from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, UsageScanner


def test_get_full_name():
  """Test get_full_name recursive flattening."""
  name_node = cst.Name("torch")
  assert get_full_name(name_node) == "torch"

  attr_node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert get_full_name(attr_node) == "torch.nn"

  attr_node_3 = cst.Attribute(
    value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")), attr=cst.Name("functional")
  )
  assert get_full_name(attr_node_3) == "torch.nn.functional"

  # Test fallback
  call_node = cst.Call(func=cst.Name("foo"))
  # Type ignore to test fallback case
  assert get_full_name(call_node) == ""  # type: ignore


def test_simple_name_scanner_found():
  """Test SimpleNameScanner when target is found."""
  scanner = SimpleNameScanner("jnp")
  code = "def foo():\n    return jnp.abs(x)\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.found is True
  assert scanner.should_traverse(module) is False


def test_simple_name_scanner_not_found():
  """Test SimpleNameScanner when target is not found."""
  scanner = SimpleNameScanner("jnp")
  code = "def foo():\n    return tf.abs(x)\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.found is False
  assert scanner.should_traverse(module) is True


def test_simple_name_scanner_inside_imports():
  """Test SimpleNameScanner ignores imports."""
  scanner = SimpleNameScanner("jnp")
  code = "import jnp\nfrom foo import jnp\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.found is False


def test_usage_scanner_import_direct():
  """Test UsageScanner tracks direct imports."""
  scanner = UsageScanner("torch")
  code = "import torch\nx = torch.abs(y)\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "torch" in scanner.found_usages


def test_usage_scanner_import_alias():
  """Test UsageScanner tracks aliased imports."""
  scanner = UsageScanner("torch")
  code = "import torch as t\nx = t.abs(y)\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "t" in scanner.found_usages


def test_usage_scanner_import_nested():
  """Test UsageScanner tracks nested imports without alias."""
  scanner = UsageScanner("torch")
  code = "import torch.nn\nx = torch.nn.Linear()\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "torch" in scanner.found_usages


def test_usage_scanner_import_from():
  """Test UsageScanner tracks from ... import ..."""
  scanner = UsageScanner("torch")
  code = "from torch import nn\nx = nn.Linear()\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_nested_alias():
  """Test UsageScanner tracks from ... import ... as ..."""
  scanner = UsageScanner("torch")
  code = "from torch.nn import Linear as L\nx = L()\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "L" in scanner.found_usages


def test_usage_scanner_import_from_no_module():
  """Test UsageScanner handles relative imports without module."""
  scanner = UsageScanner("torch")
  code = "from . import torch\nx = torch.abs()\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is True
  assert "torch" in scanner.found_usages


def test_usage_scanner_import_unrelated():
  """Test UsageScanner ignores unrelated imports."""
  scanner = UsageScanner("torch")
  code = "import os\nfrom sys import path\nx = path.join(a, b)\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is False
  assert len(scanner.found_usages) == 0


def test_usage_scanner_name_in_import_ignored():
  """Test UsageScanner ignores name usages within import statements."""
  scanner = UsageScanner("torch")
  code = "import torch as t\n"
  module = cst.parse_module(code)
  module.visit(scanner)
  assert scanner.get_result() is False
