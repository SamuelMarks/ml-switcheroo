"""Test suite for scanners coverage."""

import typing

import libcst as _cst

from ml_switcheroo.core.scanners import GlobalUsageScanner, SimpleNameScanner, get_full_name

cst: typing.Any = _cst


def test_scanners_coverage() -> None:
  """Verify scanners coverage behavior."""
  # get_full_name
  assert get_full_name(cst.Name("torch")) == "torch"
  assert get_full_name(cst.Attribute(cst.Name("torch"), cst.Name("nn"))) == "torch.nn"
  assert get_full_name(cst.Pass()) == ""

  # SimpleNameScanner
  s: typing.Any = SimpleNameScanner("target")
  tree: typing.Any = cst.parse_module("import target\nfrom x import target\nx = target")
  tree.visit(s)
  assert s.found

  # SimpleNameScanner short circuit
  s2: typing.Any = SimpleNameScanner("target")
  s2.found = True
  assert not s2.should_traverse(cst.Pass())

  # GlobalUsageScanner
  s3: typing.Any = GlobalUsageScanner()
  tree3: typing.Any = cst.parse_module("import a\nfrom b import c\nx = a + c")
  tree3.visit(s3)
  assert "a" in s3.used_names
  assert "c" in s3.used_names
  assert "x" in s3.used_names
