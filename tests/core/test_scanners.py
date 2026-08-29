"""Test suite for scanners.py."""

import typing

import libcst as cst

from ml_switcheroo.core.scanners import SimpleNameScanner, UsageScanner, get_full_name


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  module = cst.parse_module(code)
  return typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value


def test_get_full_name() -> None:
  """Docstring."""
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(parse_expr("torch.nn.functional")) == "torch.nn.functional"
  assert get_full_name(cst.Integer("42")) == ""


def test_simple_name_scanner_found() -> None:
  """Docstring."""
  module = cst.parse_module("import jnp\njnp.add(1, 2)")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert scanner.found


def test_simple_name_scanner_not_found() -> None:
  """Docstring."""
  module = cst.parse_module("import jnp\nnp.add(1, 2)")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert not scanner.found


def test_simple_name_scanner_in_import() -> None:
  """Docstring."""
  module = cst.parse_module("import jnp\nfrom jnp import x")
  scanner = SimpleNameScanner("jnp")
  module.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_basic() -> None:
  """Docstring."""
  module = cst.parse_module("import torch\ntorch.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "torch" in scanner.found_usages


def test_usage_scanner_import_as() -> None:
  """Docstring."""
  module = cst.parse_module("import torch as t\nt.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "t" in scanner.found_usages


def test_usage_scanner_import_submodule_as() -> None:
  """Docstring."""
  module = cst.parse_module("import torch.nn as nn\nnn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from() -> None:
  """Docstring."""
  module = cst.parse_module("from torch import nn\nnn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "nn" in scanner.found_usages


def test_usage_scanner_import_from_as() -> None:
  """Docstring."""
  module = cst.parse_module("from torch import nn as n\nn.Linear()")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert scanner.get_result()
  assert "n" in scanner.found_usages


def test_usage_scanner_not_used() -> None:
  """Docstring."""
  module = cst.parse_module("import torch as t\nimport numpy as np\nnp.add(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_from_not_torch() -> None:
  """Docstring."""
  module = cst.parse_module("from numpy import add\nadd(1, 2)")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


def test_usage_scanner_import_from_star() -> None:
  """Docstring."""
  module = cst.parse_module("from torch import *\n")
  scanner = UsageScanner("torch")
  module.visit(scanner)
  assert not scanner.get_result()


# --- Merged from test_scanners_gap.py ---


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


def test_simple_name_scanner_not_found_extra() -> None:
  """Verifies the behavior of simple name scanner not found."""
  code: str = "import jnp\nx = 1"
  tree: cst.Module = cst.parse_module(code)
  scanner = SimpleNameScanner("jnp")
  tree.visit(scanner)
  assert not scanner.found


def test_usage_scanner_import_from_extra() -> None:
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
  import libcst as cst

  from ml_switcheroo.core.scanners import SimpleNameScanner

  scanner = SimpleNameScanner("sys")
  scanner.found = True
  assert scanner.should_traverse(cst.Module(body=[])) is False


# --- Merged from test_scanners_extra.py ---


def test_get_full_name_extra() -> None:
  """Docstring."""
  assert get_full_name(cst.Name("x")) == "x"
  assert get_full_name(cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))) == "x.y"
  assert get_full_name(cst.Call(func=cst.Name("x"))) == ""


def test_simple_name_scanner() -> None:
  """Docstring."""
  scanner = SimpleNameScanner("target")
  mod = cst.parse_module("import target\ntarget = 1\nfrom target import x\nx = target\n")
  mod.visit(scanner)
  assert scanner.found

  scanner2 = SimpleNameScanner("target")
  mod2 = cst.parse_module("import target\nfrom target import x\n")
  mod2.visit(scanner2)
  assert not scanner2.found


def test_usage_scanner() -> None:
  """Docstring."""
  scanner = UsageScanner("torch")
  mod = cst.parse_module("import torch\ntorch.abs(x)")
  mod.visit(scanner)
  assert scanner.get_result()

  scanner2 = UsageScanner("torch")
  mod2 = cst.parse_module("import torch as t\nt.abs(x)")
  mod2.visit(scanner2)
  assert scanner2.get_result()

  scanner3 = UsageScanner("torch")
  mod3 = cst.parse_module("from torch import nn\nnn.Linear()")
  mod3.visit(scanner3)
  assert scanner3.get_result()

  scanner4 = UsageScanner("torch")
  mod4 = cst.parse_module("import torch.nn as nn\nnn.Linear()")
  mod4.visit(scanner4)
  assert scanner4.get_result()
