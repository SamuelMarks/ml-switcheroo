"""Test suite for import fixer imports mixin."""

import typing


import libcst as _cst

from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan

cst = _cst


class MockFixer(ImportMixin):
  """Test helper class MockFixer."""

  def __init__(
    self,
    plan: ResolutionPlan,
    source_fws: typing.Union[str, list[str], set[str]],
    used_names: typing.Optional[set[str]] = None,
  ) -> None:
    """Initialize test helper instance.

    Args:
        plan: Test argument or fixture.
        source_fws: Test argument or fixture.
        used_names: Test argument or fixture.
    """
    self.plan = plan
    if isinstance(source_fws, str):
      self.source_fws = {source_fws}
    elif isinstance(source_fws, list):
      self.source_fws = set(source_fws)
    else:
      self.source_fws = source_fws
    self.used_names = used_names if used_names is not None else set()
    self._satisfied_injections: set[str] = set()
    self.defined: set[str] = set()

  def _track_definition(self, node: str) -> None:  # type: ignore[override]
    """Execute track definition helper logic.

    Args:
        node: Test argument or fixture.
    """
    self.defined.add(node)


def test_make_alias_node() -> None:
  """Verify make alias node behavior."""
  fixer = MockFixer(ResolutionPlan(), set())

  # Simple req without subcomponent or alias
  req1 = ImportReq("torch")
  node1 = fixer._make_alias_node(req1)
  assert node1.asname is None
  assert isinstance(node1.name, cst.Name)
  assert node1.name.value == "torch"

  # Req with subcomponent, no alias
  req2 = ImportReq("jax", "numpy")
  node2 = fixer._make_alias_node(req2)
  assert node2.asname is None
  assert isinstance(node2.name, cst.Attribute)
  assert node2.name.attr.value == "numpy"

  # Req with alias same as leaf
  req3 = ImportReq("jax", "numpy", "numpy")
  node3 = fixer._make_alias_node(req3)
  assert node3.asname is not None
  assert isinstance(node3.asname.name, cst.Name)
  assert node3.asname.name.value == "numpy"  # dots trigger should_alias=True

  # Req with alias different from leaf
  req4 = ImportReq("jax", "numpy", "jnp")
  node4 = fixer._make_alias_node(req4)
  assert node4.asname is not None
  assert isinstance(node4.asname.name, cst.Name)
  assert node4.asname.name.value == "jnp"

  # Single module, different alias
  req5 = ImportReq("torch", alias="th")
  node5 = fixer._make_alias_node(req5)
  assert node5.asname is not None
  assert isinstance(node5.asname.name, cst.Name)
  assert node5.asname.name.value == "th"


def test_leave_import() -> None:
  """Verify leave import behavior."""
  plan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx")})
  fixer = MockFixer(plan, ["torch"], set())

  # Case: matched mapping
  original = cst.Import(names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")))])
  updated = fixer.leave_Import(original, original)

  assert isinstance(updated, cst.Import)
  assert len(updated.names) == 1
  assert updated.names[0].asname is not None
  assert isinstance(updated.names[0].asname.name, cst.Name)
  assert updated.names[0].asname.name.value == "nnx"
  assert isinstance(updated.names[0].name, cst.Attribute)
  assert updated.names[0].name.attr.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping but with preserve_alias logic
  req = ImportReq("flax")
  plan2 = ResolutionPlan(mappings={"torch": req})
  fixer2 = MockFixer(plan2, ["torch"], set())

  original_with_alias = cst.Import(
    names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("th")))]
  )
  updated_with_alias = fixer2.leave_Import(original_with_alias, original_with_alias)
  assert isinstance(updated_with_alias, cst.Import)

  # Case: unmapped import preserved if used
  fixer_used = MockFixer(ResolutionPlan(), ["torch"], {"os"})
  orig_os = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  res_os = fixer_used.leave_Import(orig_os, orig_os)
  assert isinstance(res_os, cst.Import)
  assert len(res_os.names) == 1
  assert getattr(res_os.names[0].name, "value", None) == "os"

  # Case: req with subcomponent in required_imports to hit branch 113->112
  plan_sub = ResolutionPlan(required_imports=[ImportReq("os", "path", "path")])
  fixer_sub = MockFixer(plan_sub, ["torch"], {"os"})
  res_sub = fixer_sub.leave_Import(orig_os, orig_os)
  assert isinstance(res_sub, cst.Import)
  assert updated_with_alias.names[0].asname is not None
  assert isinstance(updated_with_alias.names[0].asname.name, cst.Name)
  assert updated_with_alias.names[0].asname.name.value == "th"

  # Case: Existence check cover
  plan3 = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer3 = MockFixer(plan3, [], set())
  orig_sys = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  fixer3.leave_Import(orig_sys, orig_sys)
  assert "sys" in fixer3._satisfied_injections

  # Case: Prune (RemoveFromParent)
  plan4 = ResolutionPlan()
  fixer4 = MockFixer(plan4, ["torch"], set())
  orig_prune = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res = fixer4.leave_Import(orig_prune, orig_prune)
  assert isinstance(res, cst.RemovalSentinel)

  # Case: Used source (DCE avoids it)
  fixer5 = MockFixer(plan4, ["torch"], {"torch"})
  res2 = fixer5.leave_Import(orig_prune, orig_prune)
  assert isinstance(res2, cst.Import)

  # Case: Not in source_fws (pass through if used)
  fixer6 = MockFixer(plan4, [], {"sys"})
  orig_sys2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("sys"))])
  res3 = fixer6.leave_Import(orig_sys2, orig_sys2)
  assert isinstance(res3, cst.Import)
  assert len(res3.names) == 1


def test_leave_import_dce() -> None:
  """Verify leave import dce behavior."""
  fixer = MockFixer(
    plan=ResolutionPlan(
      mappings={},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names={"used_pkg"},
  )

  node = cst.Import(names=[cst.ImportAlias(name=cst.Name("used_pkg")), cst.ImportAlias(name=cst.Name("unused_pkg"))])

  result = fixer.leave_Import(node, node)
  assert isinstance(result, cst.Import)
  assert len(result.names) == 1
  assert result.names[0].name.value == "used_pkg"

  # Hit 121 -> 90 by having replacement occur for the FIRST alias, but NOT the second,
  # and the second is in used_names
  fixer2 = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names={"sys"},
  )
  node2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("sys"))])
  result2 = fixer2.leave_Import(node2, node2)
  assert isinstance(result2, cst.Import)
  assert len(result2.names) == 2
  assert isinstance(result2.names[0].name, cst.Attribute)
  assert result2.names[0].name.attr.value == "numpy"
  assert result2.names[1].name.value == "sys"

  # Hit the unused branch
  fixer3 = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names=set(),
  )
  node3 = cst.Import(
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
  fixer4 = MockFixer(
    plan=ResolutionPlan(
      mappings={"torch": ImportReq("jax", "numpy", "jnp")},
      required_imports=[],
    ),
    source_fws={"torch"},
    used_names=set(),
  )
  node4 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  result4 = fixer4.leave_Import(node4, node4)
  assert isinstance(result4, cst.Import)


def test_leave_import_from() -> None:
  """Verify leave import from behavior."""
  plan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx"), "torch.Tensor": ImportReq("jax", "Array")})
  fixer = MockFixer(plan, ["torch"], set())

  # Case: no module
  orig_no_mod = cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("x"))])
  assert fixer.leave_ImportFrom(orig_no_mod, orig_no_mod) is orig_no_mod

  # Case: import star source
  orig_star = cst.ImportFrom(module=cst.Name("torch"), names=cst.ImportStar())
  res_star = fixer.leave_ImportFrom(orig_star, orig_star)
  assert isinstance(res_star, cst.RemovalSentinel)

  # Case: import star non-source
  orig_star2 = cst.ImportFrom(module=cst.Name("sys"), names=cst.ImportStar())
  res_star2 = fixer.leave_ImportFrom(orig_star2, orig_star2)
  assert res_star2 is orig_star2

  # Case: matched mapping with subcomponent
  orig_nn = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res_nn = fixer.leave_ImportFrom(orig_nn, orig_nn)
  assert isinstance(res_nn, cst.Import)
  assert res_nn.names[0].asname is not None
  assert isinstance(res_nn.names[0].asname.name, cst.Name)
  assert res_nn.names[0].asname.name.value == "nnx"
  assert "flax.nnx" in fixer._satisfied_injections

  # Case: matched mapping without subcomponent
  orig_tensor = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("Tensor"))])
  res_tensor = fixer.leave_ImportFrom(orig_tensor, orig_tensor)
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
  plan2 = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", None, "flax")}, required_imports=[])
  fixer2 = MockFixer(plan2, ["torch"], set())
  node_nosub = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res5 = fixer2.leave_ImportFrom(node_nosub, node_nosub)
  assert isinstance(res5, cst.Import)
  assert res5.names[0].name.value == "flax"

  # Case: prune
  orig_other = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("optim"))])
  res_other = fixer.leave_ImportFrom(orig_other, orig_other)
  assert isinstance(res_other, cst.RemovalSentinel)

  # Case: preserve
  fixer_preserve = MockFixer(plan, ["torch"], {"optim"})
  res_preserve = fixer_preserve.leave_ImportFrom(orig_other, orig_other)
  assert isinstance(res_preserve, cst.ImportFrom)

  # Case: non-source prune (DCE removes it if unused)
  orig_sys = cst.ImportFrom(module=cst.Name("sys"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res_sys = fixer.leave_ImportFrom(orig_sys, orig_sys)
  assert isinstance(res_sys, cst.RemovalSentinel)

  # Case: from-import with asname preserved (line 183)
  orig_asname = cst.ImportFrom(
    module=cst.Name("my_mod"),
    names=[cst.ImportAlias(name=cst.Name("helper"), asname=cst.AsName(name=cst.Name("h")))],
  )
  fixer_asname = MockFixer(plan, ["torch"], {"h"})
  res_asname = fixer_asname.leave_ImportFrom(orig_asname, orig_asname)
  assert isinstance(res_asname, cst.ImportFrom)

  # Case: from-import with non-Name alias (line 187 fallback)
  orig_non_name = cst.ImportFrom(
    module=cst.Name("my_mod"),
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))],
  )
  fixer_non_name = MockFixer(plan, ["torch"], set())
  res_non_name = fixer_non_name.leave_ImportFrom(orig_non_name, orig_non_name)
  assert isinstance(res_non_name, cst.RemovalSentinel)

  # Case: Import with trailing comma on kept alias
  orig_import_comma = cst.Import(
    names=[
      cst.ImportAlias(name=cst.Name("sys"), comma=cst.Comma()),
      cst.ImportAlias(name=cst.Name("unused")),
    ]
  )
  fixer_comma = MockFixer(plan, ["torch"], {"sys"})
  res_import_comma = fixer_comma.leave_Import(orig_import_comma, orig_import_comma)
  assert isinstance(res_import_comma, cst.Import)
  assert res_import_comma.names[0].comma == cst.MaybeSentinel.DEFAULT

  # Case: ImportFrom with trailing comma on kept alias (no rpar)
  orig_from_comma = cst.ImportFrom(
    module=cst.Name("math"),
    names=[
      cst.ImportAlias(name=cst.Name("sin"), comma=cst.Comma()),
      cst.ImportAlias(name=cst.Name("unused")),
    ],
  )
  fixer_from_comma = MockFixer(plan, ["torch"], {"sin"})
  res_from_comma = fixer_from_comma.leave_ImportFrom(orig_from_comma, orig_from_comma)
  assert isinstance(res_from_comma, cst.ImportFrom)
  assert res_from_comma.names[0].comma == cst.MaybeSentinel.DEFAULT  # type: ignore[index]

  # Case: ImportFrom with rpar (parenthesized)
  orig_from_rpar = cst.ImportFrom(
    module=cst.Name("math"),
    names=[
      cst.ImportAlias(name=cst.Name("sin"), comma=cst.Comma()),
    ],
    lpar=cst.LeftParen(),
    rpar=cst.RightParen(),
  )
  fixer_rpar = MockFixer(plan, ["torch"], {"sin"})
  res_from_rpar = fixer_rpar.leave_ImportFrom(orig_from_rpar, orig_from_rpar)
  assert isinstance(res_from_rpar, cst.ImportFrom)
  assert res_from_rpar.names[0].comma != cst.MaybeSentinel.DEFAULT  # type: ignore[index]
