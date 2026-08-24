"""Test module."""

import libcst as cst
from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq


class MockFixer(ImportMixin):
  """Test element."""

  def __init__(self, plan, source_fws, preserve_source=False):
    """Test element."""
    self.plan = plan
    self.source_fws = source_fws
    self.preserve_source = preserve_source
    self._satisfied_injections = set()
    self.defined = set()

  def _track_definition(self, node):
    self.defined.add(node)


def test_make_alias_node():
  """Test element."""
  fixer = MockFixer(ResolutionPlan(), set())

  # Simple req without subcomponent or alias
  req1 = ImportReq("torch")
  node1 = fixer._make_alias_node(req1)
  assert node1.asname is None
  assert node1.name.value == "torch"

  # Req with subcomponent, no alias
  req2 = ImportReq("jax", "numpy")
  node2 = fixer._make_alias_node(req2)
  assert node2.asname is None
  assert node2.name.attr.value == "numpy"

  # Req with alias same as leaf
  req3 = ImportReq("jax", "numpy", "numpy")
  node3 = fixer._make_alias_node(req3)
  assert node3.asname is not None
  assert node3.asname.name.value == "numpy"  # dots trigger should_alias=True

  # Req with alias different from leaf
  req4 = ImportReq("jax", "numpy", "jnp")
  node4 = fixer._make_alias_node(req4)
  assert node4.asname is not None
  assert node4.asname.name.value == "jnp"

  # Single module, different alias
  req5 = ImportReq("torch", alias="th")
  node5 = fixer._make_alias_node(req5)
  assert node5.asname is not None
  assert node5.asname.name.value == "th"


def test_leave_import():
  """Test element."""
  plan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx")})
  fixer = MockFixer(plan, ["torch"], False)

  # Case: matched mapping
  original = cst.Import(names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")))])
  updated = fixer.leave_Import(original, original)

  assert isinstance(updated, cst.Import)
  assert len(updated.names) == 1
  assert updated.names[0].asname.name.value == "nnx"
  assert updated.names[0].name.attr.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping but with preserve_alias logic
  req = ImportReq("flax")
  plan2 = ResolutionPlan(mappings={"torch": req})
  fixer2 = MockFixer(plan2, ["torch"], False)

  original_with_alias = cst.Import(
    names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("th")))]
  )
  updated_with_alias = fixer2.leave_Import(original_with_alias, original_with_alias)
  assert updated_with_alias.names[0].asname.name.value == "th"

  # Case: Existence check cover
  plan3 = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer3 = MockFixer(plan3, [], False)
  orig_sys = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  fixer3.leave_Import(orig_sys, orig_sys)
  assert "sys" in fixer3._satisfied_injections

  # Case: Prune (RemoveFromParent)
  plan4 = ResolutionPlan()
  fixer4 = MockFixer(plan4, ["torch"], False)
  orig_prune = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res = fixer4.leave_Import(orig_prune, orig_prune)
  assert res == cst.RemoveFromParent()

  # Case: Preserve source
  fixer5 = MockFixer(plan4, ["torch"], True)
  res2 = fixer5.leave_Import(orig_prune, orig_prune)
  assert isinstance(res2, cst.Import)

  # Case: Not in source_fws (pass through)
  orig_sys2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  res3 = fixer5.leave_Import(orig_sys2, orig_sys2)
  assert isinstance(res3, cst.Import)
  assert len(res3.names) == 1


def test_leave_import_from():
  """Test element."""
  plan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx"), "torch.Tensor": ImportReq("jax", "Array")})
  fixer = MockFixer(plan, ["torch"], False)

  # Case: no module
  orig_no_mod = cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("x"))])
  assert fixer.leave_ImportFrom(orig_no_mod, orig_no_mod) is orig_no_mod

  # Case: import star source
  orig_star = cst.ImportFrom(module=cst.Name("torch"), names=cst.ImportStar())
  res_star = fixer.leave_ImportFrom(orig_star, orig_star)
  assert res_star == cst.RemoveFromParent()

  # Case: import star non-source
  orig_star2 = cst.ImportFrom(module=cst.Name("sys"), names=cst.ImportStar())
  res_star2 = fixer.leave_ImportFrom(orig_star2, orig_star2)
  assert res_star2 is orig_star2

  # Case: matched mapping with subcomponent
  orig_nn = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res_nn = fixer.leave_ImportFrom(orig_nn, orig_nn)
  assert isinstance(res_nn, cst.Import)
  assert res_nn.names[0].asname.name.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping without subcomponent
  orig_tensor = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("Tensor"))])
  res_tensor = fixer.leave_ImportFrom(orig_tensor, orig_tensor)
  assert isinstance(res_tensor, cst.Import)
  assert res_tensor.names[0].name.attr.value == "Array"
  assert "jax.Array" in fixer._satisfied_injections

  # Case: prune
  orig_other = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("optim"))])
  res_other = fixer.leave_ImportFrom(orig_other, orig_other)
  assert res_other == cst.RemoveFromParent()

  # Case: preserve
  fixer_preserve = MockFixer(plan, ["torch"], True)
  res_preserve = fixer_preserve.leave_ImportFrom(orig_other, orig_other)
  assert isinstance(res_preserve, cst.ImportFrom)

  # Case: non-source prune
  orig_sys = cst.ImportFrom(module=cst.Name("sys"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res_sys = fixer.leave_ImportFrom(orig_sys, orig_sys)
  assert isinstance(res_sys, cst.ImportFrom)
