"""Test suite for the Scanners Gap module."""

import libcst as cst
from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, UsageScanner


def test_get_full_name_invalid() -> None:
  """Gets full name invalid."""
  node = cst.Integer("1")
  assert get_full_name(node) == ""


def test_simple_name_scanner_import_from() -> None:
  """Verifies the behavior of simple name scanner import from."""
  code: str = "from foo import jnp\njnp.zeros(1)"
  tree: cst.Module = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_import() -> None:
  """Verifies the behavior of simple name scanner import."""
  code: str = "import jnp\njnp.zeros(1)"
  tree: cst.Module = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_not_found() -> None:
  """Verifies the behavior of simple name scanner not found."""
  code: str = "import jnp\nx = 1"
  tree: cst.Module = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_from() -> None:
  """Verifies the behavior of usage scanner import from."""
  code: str = "from torch import nn\nx = nn.Linear()"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_asname() -> None:
  """Verifies the behavior of usage scanner import from asname."""
  code: str = "from torch import nn as my_nn\nx = my_nn.Linear()"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "my_nn" in scanner.found_usages


def test_usage_scanner_import_from_other_module() -> None:
  """Verifies the behavior of usage scanner import from other module."""
  code: str = "from os import path"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_asname() -> None:
  """Verifies the behavior of usage scanner import asname."""
  code: str = "import torch as t\nx = t.abs(1)"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "t" in scanner.found_usages


def test_usage_scanner_import_other() -> None:
  """Verifies the behavior of usage scanner import other."""
  code: str = "import os\nx = os.path"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_submodule() -> None:
  """Verifies the behavior of usage scanner import submodule."""
  code: str = "import torch.nn\ntorch.nn.Linear()"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)
  assert scanner.get_result()
  assert "torch" in scanner.found_usages


def test_simple_name_scanner_should_traverse() -> None:
  """Verifies the behavior of simple name scanner should traverse."""
  code: str = "import jnp\njnp.zeros(1)\njnp.ones(1)\n"
  tree: cst.Module = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert scanner.found


def test_usage_scanner_import_from_no_module() -> None:
  """Verifies the behavior of usage scanner import from no module."""
  code: str = "from . import jax"
  tree: cst.Module = cst.parse_module(code)
  scanner = UsageScanner("jax")
  tree.visit(scanner)
  assert not scanner.get_result()
  assert "jax" not in scanner.found_usages


def test_should_traverse_optimization() -> None:
  """Verifies the behavior of should traverse optimization."""
  from ml_switcheroo.core.scanners import SimpleNameScanner
  import libcst as cst

  scanner = SimpleNameScanner("sys")
  scanner.found = True
  assert scanner.should_traverse(cst.Module(body=[])) is False
