"""Auto-generated doc."""

import libcst as cst
from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, UsageScanner


def test_get_full_name_invalid():
  """Auto-generated doc."""
  # Test with a node that is not a Name or Attribute
  node = cst.Integer("1")
  assert get_full_name(node) == ""


def test_simple_name_scanner_import_from():
  """Auto-generated doc."""
  code = "from foo import jnp\njnp.zeros(1)"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_import():
  """Auto-generated doc."""
  code = "import jnp\njnp.zeros(1)"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_not_found():
  """Auto-generated doc."""
  code = "import jnp\nx = 1"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_from():
  """Auto-generated doc."""
  code = "from torch import nn\nx = nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_asname():
  """Auto-generated doc."""
  code = "from torch import nn as my_nn\nx = my_nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "my_nn" in scanner.found_usages


def test_usage_scanner_import_from_other_module():
  """Auto-generated doc."""
  code = "from os import path"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_asname():
  """Auto-generated doc."""
  code = "import torch as t\nx = t.abs(1)"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "t" in scanner.found_usages


def test_usage_scanner_import_other():
  """Auto-generated doc."""
  code = "import os\nx = os.path"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_submodule():
  """Auto-generated doc."""
  code = "import torch.nn\ntorch.nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "torch" in scanner.found_usages


def test_simple_name_scanner_should_traverse():
  """Auto-generated doc."""
  code = "import jnp\njnp.zeros(1)\njnp.ones(1)\n"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_usage_scanner_import_from_no_module():
  """Auto-generated doc."""
  code = "from . import jax"
  tree = cst.parse_module(code)
  scanner = UsageScanner("jax")
  tree.visit(scanner)
  assert not scanner.get_result()
  assert "jax" not in scanner.found_usages


def test_should_traverse_optimization():
  """Auto-generated doc."""
  from ml_switcheroo.core.scanners import SimpleNameScanner
  import libcst as cst

  scanner = SimpleNameScanner("sys")
  scanner.found = True
  # The should_traverse hook is used to short-circuit. It returns not self.found.
  # Since found is True, it should return False.
  assert scanner.should_traverse(cst.Module(body=[])) is False
