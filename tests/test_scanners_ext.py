"""Test suite for scanners coverage with strict static typing."""

import libcst as cst

from ml_switcheroo.core.scanners import GlobalUsageScanner, SimpleNameScanner, get_full_name


def test_scanners_coverage() -> None:
  """Verify `get_full_name`, `SimpleNameScanner`, and `GlobalUsageScanner` behavior."""
  # get_full_name
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(cst.Attribute(cst.Name("torch"), cst.Name("nn"))) == "torch.nn"
  assert get_full_name(cst.Integer("1")) == ""

  # SimpleNameScanner
  s: SimpleNameScanner = SimpleNameScanner("target")
  tree: cst.Module = cst.parse_module("import target\nfrom x import target\nx = target")
  tree.visit(s)
  assert s.found is True

  # SimpleNameScanner short circuit
  s2: SimpleNameScanner = SimpleNameScanner("target")
  s2.found = True
  assert not s2.should_traverse(cst.Pass())

  # GlobalUsageScanner
  s3: GlobalUsageScanner = GlobalUsageScanner()
  tree3: cst.Module = cst.parse_module("import a\nfrom b import c\nx = a + c")
  tree3.visit(s3)
  assert "a" in s3.used_names
  assert "c" in s3.used_names
  assert "x" in s3.used_names
