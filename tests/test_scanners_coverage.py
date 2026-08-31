"""Docstring."""

import libcst as cst
from ml_switcheroo.core.scanners import get_full_name, SimpleNameScanner, GlobalUsageScanner


def test_scanners_coverage() -> None:
  """Docstring."""
  # get_full_name
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(cst.Attribute(cst.Name("torch"), cst.Name("nn"))) == "torch.nn"
  assert get_full_name(cst.Pass()) == ""

  # SimpleNameScanner
  s = SimpleNameScanner("target")
  tree = cst.parse_module("import target\nfrom x import target\nx = target")
  tree.visit(s)
  assert s.found

  # SimpleNameScanner short circuit
  s2 = SimpleNameScanner("target")
  s2.found = True
  assert not s2.should_traverse(cst.Pass())

  # GlobalUsageScanner
  s3 = GlobalUsageScanner()
  tree3 = cst.parse_module("import a\nfrom b import c\nx = a + c")
  tree3.visit(s3)
  assert "a" in s3.used_names
  assert "c" in s3.used_names
  assert "x" in s3.used_names
