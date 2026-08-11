"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, UsageScanner


def test_get_full_name():
  """Test get full name."""
  assert get_full_name(cst.Name("x")) == "x"
  assert get_full_name(cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))) == "x.y"
  assert get_full_name(cst.Call(func=cst.Name("x"))) == ""


def test_simple_name_scanner():
  """Test simple name scanner."""
  scanner = SimpleNameScanner("target")
  mod = cst.parse_module("import target\ntarget = 1\nfrom target import x\nx = target\n")
  mod.visit(scanner)
  assert scanner.found

  scanner2 = SimpleNameScanner("target")
  mod2 = cst.parse_module("import target\nfrom target import x\n")
  mod2.visit(scanner2)
  assert not scanner2.found


def test_usage_scanner():
  """Test usage scanner."""
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
