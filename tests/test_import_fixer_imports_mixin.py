"""Test module."""

from typing import List, Set, Union

import libcst as cst

from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


class MockFixer(ImportMixin):
  """Docstring."""

  def __init__(
    self, plan: ResolutionPlan, source_fws: Union[str, List[str], Set[str]], preserve_source: bool = False
  ) -> None:
    """Docstring."""
    self.plan: ResolutionPlan = plan
    self.source_fws: Union[str, List[str], Set[str]] = source_fws
    self.preserve_source: bool = preserve_source
    self._satisfied_injections: Set[str] = set()
    self.defined: Set[cst.CSTNode] = set()

  def _track_definition(self, node: cst.CSTNode) -> None:
    self.defined.add(node)


def test_make_alias_node() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(ResolutionPlan(), set())

  # Simple req without subcomponent or alias
  req1: ImportReq = ImportReq("torch")
  node1: cst.ImportAlias = fixer._make_alias_node(req1)
  assert node1.asname is None
  assert isinstance(node1.name, cst.Name)
  assert node1.name.value == "torch"

  # Req with subcomponent, no alias
  req2: ImportReq = ImportReq("jax", "numpy")
  node2: cst.ImportAlias = fixer._make_alias_node(req2)
  assert node2.asname is None
  assert isinstance(node2.name, cst.Attribute)
  assert node2.name.attr.value == "numpy"

  # Req with alias same as leaf
  req3: ImportReq = ImportReq("jax", "numpy", "numpy")
  node3: cst.ImportAlias = fixer._make_alias_node(req3)
  assert node3.asname is not None
  assert isinstance(node3.asname.name, cst.Name)
  assert node3.asname.name.value == "numpy"  # dots trigger should_alias=True

  # Req with alias different from leaf
  req4: ImportReq = ImportReq("jax", "numpy", "jnp")
  node4: cst.ImportAlias = fixer._make_alias_node(req4)
  assert node4.asname is not None
  assert isinstance(node4.asname.name, cst.Name)
  assert node4.asname.name.value == "jnp"

  # Single module, different alias
  req5: ImportReq = ImportReq("torch", alias="th")
  node5: cst.ImportAlias = fixer._make_alias_node(req5)
  assert node5.asname is not None
  assert isinstance(node5.asname.name, cst.Name)
  assert node5.asname.name.value == "th"


def test_leave_import() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx")})
  fixer: MockFixer = MockFixer(plan, ["torch"], False)

  # Case: matched mapping
  original: cst.Import = cst.Import(
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")))]
  )
  updated: Union[cst.Import, cst.RemovalSentinel] = fixer.leave_Import(original, original)

  assert isinstance(updated, cst.Import)
  assert len(updated.names) == 1
  assert updated.names[0].asname is not None
  assert isinstance(updated.names[0].asname.name, cst.Name)
  assert updated.names[0].asname.name.value == "nnx"
  assert isinstance(updated.names[0].name, cst.Attribute)
  assert updated.names[0].name.attr.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping but with preserve_alias logic
  req: ImportReq = ImportReq("flax")
  plan2: ResolutionPlan = ResolutionPlan(mappings={"torch": req})
  fixer2: MockFixer = MockFixer(plan2, ["torch"], False)

  original_with_alias: cst.Import = cst.Import(
    names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("th")))]
  )
  updated_with_alias: Union[cst.Import, cst.RemovalSentinel] = fixer2.leave_Import(
    original_with_alias, original_with_alias
  )
  assert isinstance(updated_with_alias, cst.Import)
  assert updated_with_alias.names[0].asname is not None
  assert isinstance(updated_with_alias.names[0].asname.name, cst.Name)
  assert updated_with_alias.names[0].asname.name.value == "th"

  # Case: Existence check cover
  plan3: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer3: MockFixer = MockFixer(plan3, [], False)
  orig_sys: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  fixer3.leave_Import(orig_sys, orig_sys)
  assert "sys" in fixer3._satisfied_injections

  # Case: Prune (RemoveFromParent)
  plan4: ResolutionPlan = ResolutionPlan()
  fixer4: MockFixer = MockFixer(plan4, ["torch"], False)
  orig_prune: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res: Union[cst.Import, cst.RemovalSentinel] = fixer4.leave_Import(orig_prune, orig_prune)
  assert isinstance(res, cst.RemovalSentinel)

  # Case: Preserve source
  fixer5: MockFixer = MockFixer(plan4, ["torch"], True)
  res2: Union[cst.Import, cst.RemovalSentinel] = fixer5.leave_Import(orig_prune, orig_prune)
  assert isinstance(res2, cst.Import)

  # Case: Not in source_fws (pass through)
  orig_sys2: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  res3: Union[cst.Import, cst.RemovalSentinel] = fixer5.leave_Import(orig_sys2, orig_sys2)
  assert isinstance(res3, cst.Import)
  assert len(res3.names) == 1


def test_leave_import_from() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan(
    mappings={"torch.nn": ImportReq("flax", "nnx", "nnx"), "torch.Tensor": ImportReq("jax", "Array")}
  )
  fixer: MockFixer = MockFixer(plan, ["torch"], False)

  # Case: no module
  orig_no_mod: cst.ImportFrom = cst.ImportFrom(
    module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("x"))]
  )
  assert fixer.leave_ImportFrom(orig_no_mod, orig_no_mod) is orig_no_mod

  # Case: import star source
  orig_star: cst.ImportFrom = cst.ImportFrom(module=cst.Name("torch"), names=cst.ImportStar())
  res_star: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_star, orig_star)
  assert isinstance(res_star, cst.RemovalSentinel)

  # Case: import star non-source
  orig_star2: cst.ImportFrom = cst.ImportFrom(module=cst.Name("sys"), names=cst.ImportStar())
  res_star2: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_star2, orig_star2)
  assert res_star2 is orig_star2

  # Case: matched mapping with subcomponent
  orig_nn: cst.ImportFrom = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res_nn: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_nn, orig_nn)
  assert isinstance(res_nn, cst.Import)
  assert res_nn.names[0].asname is not None
  assert isinstance(res_nn.names[0].asname.name, cst.Name)
  assert res_nn.names[0].asname.name.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping without subcomponent
  orig_tensor: cst.ImportFrom = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("Tensor"))])
  res_tensor: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_tensor, orig_tensor)
  assert isinstance(res_tensor, cst.Import)
  assert isinstance(res_tensor.names[0].name, cst.Attribute)
  assert res_tensor.names[0].name.attr.value == "Array"
  assert "jax.Array" in fixer._satisfied_injections

  # Case: prune
  orig_other: cst.ImportFrom = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("optim"))])
  res_other: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_other, orig_other)
  assert isinstance(res_other, cst.RemovalSentinel)

  # Case: preserve
  fixer_preserve: MockFixer = MockFixer(plan, ["torch"], True)
  res_preserve: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer_preserve.leave_ImportFrom(
    orig_other, orig_other
  )
  assert isinstance(res_preserve, cst.ImportFrom)

  # Case: non-source prune
  orig_sys: cst.ImportFrom = cst.ImportFrom(module=cst.Name("sys"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res_sys: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_sys, orig_sys)
  assert isinstance(res_sys, cst.ImportFrom)
