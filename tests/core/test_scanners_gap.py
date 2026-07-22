"""Test suite for the Scanners Gap module."""

import libcst as cst
from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, UsageScanner


def test_get_full_name_invalid():
  """Gets full name invalid."""
  node = cst.Integer("1")
  assert get_full_name(node) == ""


def test_simple_name_scanner_import_from():
  """Verifies the behavior of simple name scanner import from."""
  code = "from foo import jnp\njnp.zeros(1)"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_import():
  """Verifies the behavior of simple name scanner import."""
  code = "import jnp\njnp.zeros(1)"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_not_found():
  """Verifies the behavior of simple name scanner not found."""
  code = "import jnp\nx = 1"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_from():
  """Verifies the behavior of usage scanner import from."""
  code = "from torch import nn\nx = nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_asname():
  """Verifies the behavior of usage scanner import from asname."""
  code = "from torch import nn as my_nn\nx = my_nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "my_nn" in scanner.found_usages


def test_usage_scanner_import_from_other_module():
  """Verifies the behavior of usage scanner import from other module."""
  code = "from os import path"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_asname():
  """Verifies the behavior of usage scanner import asname."""
  code = "import torch as t\nx = t.abs(1)"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "t" in scanner.found_usages


def test_usage_scanner_import_other():
  """Verifies the behavior of usage scanner import other."""
  code = "import os\nx = os.path"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_submodule():
  """Verifies the behavior of usage scanner import submodule."""
  code = "import torch.nn\ntorch.nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "torch" in scanner.found_usages


def test_simple_name_scanner_should_traverse():
  """Verifies the behavior of simple name scanner should traverse."""
  code = "import jnp\njnp.zeros(1)\njnp.ones(1)\n"
  tree = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_usage_scanner_import_from_no_module():
  """Verifies the behavior of usage scanner import from no module."""
  code = "from . import jax"
  tree = cst.parse_module(code)
  scanner = UsageScanner("jax")
  tree.visit(scanner)
  assert not scanner.get_result()
  assert "jax" not in scanner.found_usages


def test_should_traverse_optimization():
  """Verifies the behavior of should traverse optimization."""
  from ml_switcheroo.core.scanners import SimpleNameScanner
  import libcst as cst

  scanner = SimpleNameScanner("sys")
  scanner.found = True
  assert scanner.should_traverse(cst.Module(body=[])) is False
