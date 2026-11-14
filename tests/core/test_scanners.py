"""Unit tests for AST symbol usage scanners (`ml_switcheroo.core.scanners`)."""

import libcst as cst

from ml_switcheroo.core.scanners import GlobalUsageScanner, SimpleNameScanner, get_full_name

_NL = chr(10)


def test_get_full_name_and_simple_name_scanner() -> None:
  """Verify `get_full_name` dotted resolution and `SimpleNameScanner` short-circuit traversal."""
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(cst.Attribute(cst.Name("torch"), cst.Name("nn"))) == "torch.nn"
  assert get_full_name(cst.Integer("1")) == ""

  scanner = SimpleNameScanner("target")
  tree = cst.parse_module(_NL.join(["import target", "from x import target", "x = target"]))
  tree.visit(scanner)
  assert scanner.found is True

  already_found = SimpleNameScanner("target")
  already_found.found = True
  assert already_found.should_traverse(cst.Pass()) is False


def test_global_usage_scanner_attributes_and_imports() -> None:
  """Verify `GlobalUsageScanner` records referenced root names while ignoring unused import aliases."""
  unused_alias = cst.parse_module(_NL.join(["import torch.nn as nn", "self.conv = keras.layers.Conv2D(32, 3)"]))
  s1 = GlobalUsageScanner()
  unused_alias.visit(s1)
  assert "nn" not in s1.used_names
  assert "keras" in s1.used_names

  used_alias = cst.parse_module(_NL.join(["import torch.nn as nn", "self.conv = nn.Conv2D(32, 3)"]))
  s2 = GlobalUsageScanner()
  used_alias.visit(s2)
  assert "nn" in s2.used_names

  nested_attr = cst.parse_module(_NL.join(["import torch", "from b import c", "x = torch.nn.Conv2D(32, 3) + c"]))
  s3 = GlobalUsageScanner()
  nested_attr.visit(s3)
  assert {"torch", "c", "x"} <= s3.used_names
