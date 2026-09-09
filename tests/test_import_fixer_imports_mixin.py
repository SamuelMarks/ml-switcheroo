"""Test module."""

from typing import List, Optional, Set, Union

import libcst as cst

from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


class MockFixer(ImportMixin):
  """Docstring."""

  def __init__(
    self, plan: ResolutionPlan, source_fws: Union[str, List[str], Set[str]], used_names: Optional[Set[str]] = None
  ) -> None:
    """Docstring."""
    self.plan: ResolutionPlan = plan
    if isinstance(source_fws, str):
      self.source_fws = {source_fws}
    elif isinstance(source_fws, list):
      self.source_fws = set(source_fws)
    else:
      self.source_fws = source_fws
    self.used_names: Set[str] = used_names if used_names is not None else set()
    self._satisfied_injections: Set[str] = set()
    self.defined: Set[cst.CSTNode] = set()

  def _track_definition(self, node: cst.CSTNode) -> None:
    """Docstring."""
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
  fixer: MockFixer = MockFixer(plan, ["torch"], set())

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
  fixer2: MockFixer = MockFixer(plan2, ["torch"], set())

  original_with_alias: cst.Import = cst.Import(
    names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("th")))]
  )
  updated_with_alias: Union[cst.Import, cst.RemovalSentinel] = fixer2.leave_Import(
    original_with_alias, original_with_alias
  )
  assert isinstance(updated_with_alias, cst.Import)

  # Case: unmapped import preserved if used
  fixer_used: MockFixer = MockFixer(ResolutionPlan(), ["torch"], {"os"})
  orig_os: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  res_os: Union[cst.Import, cst.RemovalSentinel] = fixer_used.leave_Import(orig_os, orig_os)
  assert isinstance(res_os, cst.Import)
  assert len(res_os.names) == 1
  assert getattr(res_os.names[0].name, "value", None) == "os"

  # Case: req with subcomponent in required_imports to hit branch 113->112
  plan_sub: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("os", "path", "path")])
  fixer_sub: MockFixer = MockFixer(plan_sub, ["torch"], {"os"})
  res_sub: Union[cst.Import, cst.RemovalSentinel] = fixer_sub.leave_Import(orig_os, orig_os)
  assert isinstance(res_sub, cst.Import)
  assert updated_with_alias.names[0].asname is not None
  assert isinstance(updated_with_alias.names[0].asname.name, cst.Name)
  assert updated_with_alias.names[0].asname.name.value == "th"

  # Case: Existence check cover
  plan3: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer3: MockFixer = MockFixer(plan3, [], set())
  orig_sys: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  fixer3.leave_Import(orig_sys, orig_sys)
  assert "sys" in fixer3._satisfied_injections

  # Case: Prune (RemoveFromParent)
  plan4: ResolutionPlan = ResolutionPlan()
  fixer4: MockFixer = MockFixer(plan4, ["torch"], set())
  orig_prune: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res: Union[cst.Import, cst.RemovalSentinel] = fixer4.leave_Import(orig_prune, orig_prune)
  assert isinstance(res, cst.RemovalSentinel)

  # Case: Used source (DCE avoids it)
  fixer5: MockFixer = MockFixer(plan4, ["torch"], {"torch"})
  res2: Union[cst.Import, cst.RemovalSentinel] = fixer5.leave_Import(orig_prune, orig_prune)
  assert isinstance(res2, cst.Import)

  # Case: Not in source_fws (pass through if used)
  fixer6: MockFixer = MockFixer(plan4, [], {"sys"})
  orig_sys2: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  res3: Union[cst.Import, cst.RemovalSentinel] = fixer6.leave_Import(orig_sys2, orig_sys2)
  assert isinstance(res3, cst.Import)
  assert len(res3.names) == 1


def test_leave_import_dce() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(
      mappings={},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names={"used_pkg"},
  )

  node: cst.Import = cst.Import(
    names=[cst.ImportAlias(name=cst.Name("used_pkg")), cst.ImportAlias(name=cst.Name("unused_pkg"))]
  )

  result: Union[cst.Import, cst.RemovalSentinel] = fixer.leave_Import(node, node)
  assert isinstance(result, cst.Import)
  assert len(result.names) == 1
  assert result.names[0].name.value == "used_pkg"

  # Hit 121 -> 90 by having replacement occur for the FIRST alias, but NOT the second,
  # and the second is in used_names
  fixer2: MockFixer = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names={"sys"},
  )
  node2: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("sys"))])
  result2 = fixer2.leave_Import(node2, node2)
  assert isinstance(result2, cst.Import)
  assert len(result2.names) == 2
  assert isinstance(result2.names[0].name, cst.Attribute)
  assert result2.names[0].name.attr.value == "numpy"
  assert result2.names[1].name.value == "sys"

  # Hit the unused branch
  fixer3: MockFixer = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names=set(),
  )
  node3: cst.Import = cst.Import(
    names=[
      cst.ImportAlias(name=cst.Name("torch")),
      cst.ImportAlias(name=cst.Name("sys")),
      cst.ImportAlias(name=cst.Name("os")),
    ]
  )
  result3 = fixer3.leave_Import(node3, node3)
  assert isinstance(result3, cst.Import)
  assert len(result3.names) == 1

  # Hit replacement_occurred=True jumping back to loop start (line 121->89)
  fixer4: MockFixer = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names=set(),
  )
  node4: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  result4 = fixer4.leave_Import(node4, node4)
  assert isinstance(result4, cst.Import)


def test_leave_import_from() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan(
    mappings={"torch.nn": ImportReq("flax", "nnx", "nnx"), "torch.Tensor": ImportReq("jax", "Array")}
  )
  fixer: MockFixer = MockFixer(plan, ["torch"], set())

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

  # Multiple aliases (bypasses 160)
  node_multiple = cst.ImportFrom(
    module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("a")), cst.ImportAlias(name=cst.Name("b"))]
  )
  res4 = fixer.leave_ImportFrom(node_multiple, node_multiple)
  assert isinstance(res4, cst.RemovalSentinel)

  # Hit 175-179 branch: matching mapping with NO subcomponent
  plan2: ResolutionPlan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", None, "flax")}, required_imports=[])
  fixer2: MockFixer = MockFixer(plan2, ["torch"], set())
  node_nosub = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res5 = fixer2.leave_ImportFrom(node_nosub, node_nosub)
  assert isinstance(res5, cst.Import)
  assert res5.names[0].name.value == "flax"

  # Case: prune
  orig_other: cst.ImportFrom = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("optim"))])
  res_other: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_other, orig_other)
  assert isinstance(res_other, cst.RemovalSentinel)

  # Case: preserve
  fixer_preserve: MockFixer = MockFixer(plan, ["torch"], {"optim"})
  res_preserve: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer_preserve.leave_ImportFrom(
    orig_other, orig_other
  )
  assert isinstance(res_preserve, cst.ImportFrom)

  # Case: non-source prune (DCE removes it if unused)
  orig_sys: cst.ImportFrom = cst.ImportFrom(module=cst.Name("sys"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res_sys: Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel] = fixer.leave_ImportFrom(orig_sys, orig_sys)
  assert isinstance(res_sys, cst.RemovalSentinel)

  # Case: from-import with asname preserved (line 183)
  orig_asname: cst.ImportFrom = cst.ImportFrom(
    module=cst.Name("my_mod"),
    names=[cst.ImportAlias(name=cst.Name("helper"), asname=cst.AsName(name=cst.Name("h")))],
  )
  fixer_asname: MockFixer = MockFixer(plan, ["torch"], {"h"})
  res_asname = fixer_asname.leave_ImportFrom(orig_asname, orig_asname)
  assert isinstance(res_asname, cst.ImportFrom)

  # Case: from-import with non-Name alias (line 187 fallback)
  orig_non_name: cst.ImportFrom = cst.ImportFrom(
    module=cst.Name("my_mod"),
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))],
  )
  fixer_non_name: MockFixer = MockFixer(plan, ["torch"], set())
  res_non_name = fixer_non_name.leave_ImportFrom(orig_non_name, orig_non_name)
  assert isinstance(res_non_name, cst.RemovalSentinel)
