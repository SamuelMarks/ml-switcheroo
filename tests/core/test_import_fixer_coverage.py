import pytest
import libcst as cst
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan


def test_attributes_mixin_missing():
  plan = ResolutionPlan()
  fixer = ImportFixer(plan)

  fixer.target_fw = "nnx"
  node_to_simplify = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")), attr=cst.Name("Module")
  )
  res = fixer.leave_Attribute(node_to_simplify, node_to_simplify)
  assert res is not node_to_simplify

  node_no_name = cst.Attribute(value=cst.Call(func=cst.Name("a")), attr=cst.Name("b"))
  res_no_name = fixer.leave_Attribute(node_no_name, node_no_name)
  assert res_no_name == node_no_name

  fixer._path_to_alias = None
  node_valid = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  res_valid = fixer.leave_Attribute(node_valid, node_valid)
  assert res_valid == node_valid
  fixer._path_to_alias = {}

  node_unsafe = cst.Attribute(
    value=cst.Attribute(value=cst.Name("unsafe_root"), attr=cst.Name("module")), attr=cst.Name("Module")
  )
  res_unsafe = fixer.leave_Attribute(node_unsafe, node_unsafe)
  assert res_unsafe == node_unsafe

  fixer._defined_names = {"foo"}
  fixer._path_to_alias = {"foo.bar": "bar"}
  fixer.target_fw = "baz"
  node_safe = cst.Attribute(value=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("module")), attr=cst.Name("Module"))
  res_safe = fixer.leave_Attribute(node_safe, node_safe)
  assert res_safe is not node_safe

  # 52-61
  fixer._path_to_alias = {"a.b": "c"}
  node_collapse = cst.Attribute(value=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")), attr=cst.Name("c"))
  res_collapse = fixer.leave_Attribute(node_collapse, node_collapse)
  assert res_collapse is not node_collapse


def test_attributes_mixin_no_full_name():
  plan = ResolutionPlan()
  fixer = ImportFixer(plan)

  # get_full_name returns None if the node has things like Calls
  node = cst.Attribute(value=cst.Call(func=cst.Name("a")), attr=cst.Name("b"))
  res = fixer.leave_Attribute(node, node)
  assert res == node


def test_imports_mixin_alias():
  plan = ResolutionPlan()
  fixer = ImportFixer(plan)
  fixer._path_to_alias = {"foo": "bar"}

  # 99: module missing
  node_empty = cst.ImportFrom(module=None, names=cst.ImportStar(), relative=[cst.Dot()])
  assert fixer.leave_ImportFrom(node_empty, node_empty) == node_empty

  # 106: remove from parent
  fixer.source_fws = ["foo"]
  node_star = cst.ImportFrom(module=cst.Name("foo"), names=cst.ImportStar())
  res_star = fixer.leave_ImportFrom(node_star, node_star)
  assert isinstance(res_star, type(cst.RemoveFromParent()))


def test_base_import_fixer():
  from ml_switcheroo.core.import_fixer.base import BaseImportFixer

  # base has lines 44, 70
  pass


def test_imports_mixin_107():
  plan = ResolutionPlan()
  fixer = ImportFixer(plan)
  fixer.source_fws = ["foo"]
  fixer.preserve_source = True
  node_star = cst.ImportFrom(module=cst.Name("foo"), names=cst.ImportStar())
  res_star = fixer.leave_ImportFrom(node_star, node_star)
  assert res_star == node_star


def test_base_import_fixer():
  from ml_switcheroo.core.import_fixer.base import BaseImportFixer

  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan, source_fws="torch")
  assert "torch" in fixer.source_fws

  fixer2 = BaseImportFixer(plan, source_fws=["jax"])
  assert "jax" in fixer2.source_fws

  alias1 = cst.ImportAlias(name=cst.Name("a"), asname=cst.AsName(name=cst.Name("b")))
  fixer2._track_definition(alias1)
  assert "b" in fixer2._defined_names

  alias2 = cst.ImportAlias(name=cst.Name("c"), asname=cst.AsName(name=cst.Name("d")))
  fixer2._track_definition(alias2)
  assert "d" in fixer2._defined_names

  alias3 = cst.ImportAlias(name=cst.Attribute(value=cst.Name("e"), attr=cst.Name("f")))
  fixer2._track_definition(alias3)
  assert "e" in fixer2._defined_names


def test_attributes_mixin_line_40():
  plan = ResolutionPlan()
  fixer = ImportFixer(plan)

  # get_full_name returns None
  node = cst.Attribute(value=cst.Call(func=cst.Name("a")), attr=cst.Name("b"))
  res = fixer.leave_Attribute(node, node)
  assert res == node
