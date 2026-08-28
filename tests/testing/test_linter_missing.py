"""Test suite for the Linter Missing module."""

from typing import List


def test_linter_missing_coverage() -> None:
  """Verifies the behavior of linter missing coverage."""
  from ml_switcheroo.testing.linter import StructuralLinter

  linter: StructuralLinter = StructuralLinter({"torch"})
  res: List[str] = linter.check("from torch import *")
  assert any(("Wildcard" in msg for msg in res))
  import libcst as cst

  assert linter._get_root_name(cst.Integer("1")) == ""
  assert linter._get_full_name_from_node(cst.Integer("1")) == ""


def test_linter_get_full_name_attribute() -> None:
  """Verifies the behavior of linter get full name attribute."""
  from ml_switcheroo.testing.linter import StructuralLinter
  import libcst as cst

  linter: StructuralLinter = StructuralLinter({"torch"})
  node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert linter._get_full_name_from_node(node) == "torch.nn"


def test_linter_parse_error() -> None:
  """Verifies the behavior of linter parse correctly handling an error."""
  from ml_switcheroo.testing.linter import validate_transpilation

  ok: bool
  msgs: List[str]
  ok, msgs = validate_transpilation("def foo(", "torch")
  assert not ok
  assert any(("Parse Error" in m for m in msgs))


def test_linter_direct_access() -> None:
  """Verifies the behavior of linter direct access."""
  from ml_switcheroo.testing.linter import validate_transpilation

  code: str = "import something_else\ntorch.add(x, y)"
  ok: bool
  msgs: List[str]
  ok, msgs = validate_transpilation(code, "torch")
  assert not ok
  assert any(("Direct access 'torch'" in m for m in msgs))
