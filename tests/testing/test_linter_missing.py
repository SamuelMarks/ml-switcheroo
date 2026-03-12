def test_linter_missing_coverage():
  from ml_switcheroo.testing.linter import StructuralLinter

  linter = StructuralLinter({"torch"})

  # 116: Forbidden wildcard
  res = linter.check("from torch import *")
  assert any("Wildcard" in msg for msg in res)

  # 192: Extract root name string
  import libcst as cst

  assert linter._get_root_name(cst.Integer("1")) == ""

  # 200-202: Full name from node unknown type
  assert linter._get_full_name_from_node(cst.Integer("1")) == ""


def test_linter_get_full_name_attribute():
  from ml_switcheroo.testing.linter import StructuralLinter
  import libcst as cst

  linter = StructuralLinter({"torch"})
  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert linter._get_full_name_from_node(node) == "torch.nn"


def test_linter_parse_error():
  from ml_switcheroo.testing.linter import validate_transpilation

  # 57-58: Linter Parse Error
  ok, msgs = validate_transpilation("def foo(", "torch")
  assert not ok
  assert any("Parse Error" in m for m in msgs)


def test_linter_direct_access():
  from ml_switcheroo.testing.linter import validate_transpilation

  # 156-159: Direct access of forbidden roots without alias
  code = "import something_else\ntorch.add(x, y)"
  ok, msgs = validate_transpilation(code, "torch")
  assert not ok
  assert any("Direct access 'torch'" in m for m in msgs)
