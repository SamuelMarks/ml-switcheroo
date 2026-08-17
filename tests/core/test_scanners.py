"""Test suite for scanners.py."""

import libcst as cst

from ml_switcheroo.core.scanners import (
  get_full_name,
  SimpleNameScanner,
  UsageScanner,
)


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  module = cst.parse_module(code)
  return module.body[0].body[0].value


def test_get_full_name():
  """Docstring."""
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(parse_expr("torch.nn.functional")) == "torch.nn.functional"
  assert get_full_name(cst.Integer("42")) == ""


def test_simple_name_scanner_found():
  """Docstring."""
  module = cst.parse_module("import jnp\njnp.add(1, 2)")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_not_found():
  """Docstring."""
  module = cst.parse_module("import jnp\nnp.add(1, 2)")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert not scanner.found


def test_simple_name_scanner_in_import():
  """Docstring."""
  module = cst.parse_module("import jnp\nfrom jnp import x")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_basic():
  """Docstring."""
  module = cst.parse_module("import torch\ntorch.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "torch" in scanner.found_usages


def test_usage_scanner_import_as():
  """Docstring."""
  module = cst.parse_module("import torch as t\nt.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "t" in scanner.found_usages


def test_usage_scanner_import_submodule_as():
  """Docstring."""
  module = cst.parse_module("import torch.nn as nn\nnn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from():
  """Docstring."""
  module = cst.parse_module("from torch import nn\nnn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_as():
  """Docstring."""
  module = cst.parse_module("from torch import nn as n\nn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "n" in scanner.found_usages


def test_usage_scanner_not_used():
  """Docstring."""
  module = cst.parse_module("import torch as t\nimport numpy as np\nnp.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_from_not_torch():
  """Docstring."""
  module = cst.parse_module("from numpy import add\nadd(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_from_star():
  """Docstring."""
  module = cst.parse_module("from torch import *\n")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()
