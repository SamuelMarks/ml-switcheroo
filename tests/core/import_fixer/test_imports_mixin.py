"""Unit tests for ``ImportStatementRewriter`` in the import fixer module."""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.imports_mixin import ImportStatementRewriter
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan
from ml_switcheroo.core.scanners import get_full_name


def _make_rewriter(
  plan: ResolutionPlan | None = None,
  source_frameworks: str | typing.Iterable[str] | None = None,
  used_names: set[str] | None = None,
) -> ImportStatementRewriter:
  """Construct an ``ImportStatementRewriter`` backed by ``ImportFixerState``.

  Args:
      plan: Optional ``ResolutionPlan`` (defaults to empty plan).
      source_frameworks: Framework root name(s) eligible for source pruning/mapping.
      used_names: Identifier names referenced in the module body for dead-code elimination.

  Returns:
      Initialized ``ImportStatementRewriter`` instance.
  """
  active_plan = plan if plan is not None else ResolutionPlan()
  state = ImportFixerState.create(
    plan=active_plan,
    source_frameworks=source_frameworks,
    used_names=used_names,
  )
  return ImportStatementRewriter(state)


def _parse_import(stmt_code: str) -> cst.Import:
  """Parse a Python ``import ...`` statement string into a ``cst.Import`` node.

  Args:
      stmt_code: Python import statement source string.

  Returns:
      Parsed ``cst.Import`` CST node.
  """
  stmt = cst.parse_statement(stmt_code)
  assert isinstance(stmt, cst.SimpleStatementLine)
  imp = stmt.body[0]
  assert isinstance(imp, cst.Import)
  return imp


def _parse_import_from(stmt_code: str) -> cst.ImportFrom:
  """Parse a Python ``from ... import ...`` statement string into a ``cst.ImportFrom`` node.

  Args:
      stmt_code: Python from-import statement source string.

  Returns:
      Parsed ``cst.ImportFrom`` CST node.
  """
  stmt = cst.parse_statement(stmt_code)
  assert isinstance(stmt, cst.SimpleStatementLine)
  imp = stmt.body[0]
  assert isinstance(imp, cst.ImportFrom)
  return imp


def test_make_alias_node_variants() -> None:
  """Verify ``make_alias_node`` handles plain modules, dotted submodules, and redundant vs explicit aliases."""
  rewriter = _make_rewriter()

  node_plain = rewriter.make_alias_node(ImportReq("torch"))
  assert node_plain.asname is None
  assert isinstance(node_plain.name, cst.Name)
  assert node_plain.name.value == "torch"

  node_empty_alias = rewriter.make_alias_node(ImportReq(module="jax", alias=""))
  assert node_empty_alias.asname is None

  node_same_root_alias = rewriter.make_alias_node(ImportReq(module="jax", alias="jax"))
  assert node_same_root_alias.asname is None

  node_sub_no_alias = rewriter.make_alias_node(ImportReq("jax", "numpy"))
  assert node_sub_no_alias.asname is None
  assert isinstance(node_sub_no_alias.name, cst.Attribute)
  assert node_sub_no_alias.name.attr.value == "numpy"

  node_sub_same_leaf = rewriter.make_alias_node(ImportReq("jax", "numpy", "numpy"))
  assert node_sub_same_leaf.asname is not None
  assert isinstance(node_sub_same_leaf.asname.name, cst.Name)
  assert node_sub_same_leaf.asname.name.value == "numpy"

  node_dotted_same_leaf = rewriter.make_alias_node(ImportReq(module="jax.numpy", alias="numpy"))
  assert node_dotted_same_leaf.asname is not None
  assert isinstance(node_dotted_same_leaf.asname.name, cst.Name)
  assert node_dotted_same_leaf.asname.name.value == "numpy"

  node_sub_diff_alias = rewriter.make_alias_node(ImportReq("jax", "numpy", "jnp"))
  assert node_sub_diff_alias.asname is not None
  assert isinstance(node_sub_diff_alias.asname.name, cst.Name)
  assert node_sub_diff_alias.asname.name.value == "jnp"

  node_root_diff_alias = rewriter.make_alias_node(ImportReq("torch", alias="th"))
  assert node_root_diff_alias.asname is not None
  assert isinstance(node_root_diff_alias.asname.name, cst.Name)
  assert node_root_diff_alias.asname.name.value == "th"


def test_rewrite_import_mapped_and_alias_preservation() -> None:
  """Verify ``rewrite_import`` maps source imports and preserves user aliases when target alias is unset."""
  plan = ResolutionPlan(mappings={"torch.nn": ImportReq("flax", "nnx", "nnx")})
  rewriter = _make_rewriter(plan=plan, source_frameworks=["torch"])

  node = _parse_import("import torch.nn")
  updated = rewriter.rewrite_import(node, node)
  assert isinstance(updated, cst.Import)
  assert len(updated.names) == 1
  assert isinstance(updated.names[0].name, cst.Attribute)
  assert updated.names[0].name.attr.value == "nnx"
  assert updated.names[0].asname is not None
  assert isinstance(updated.names[0].asname.name, cst.Name)
  assert updated.names[0].asname.name.value == "nnx"
  assert "flax.nnx" in rewriter.state.satisfied_injections
  assert "nnx" in rewriter.state.defined_names

  plan_preserve = ResolutionPlan(
    mappings={
      "torch": ImportReq("flax"),
      "torch.sub": ImportReq(module="jax", subcomponent="sub"),
    }
  )
  rewriter_preserve = _make_rewriter(plan=plan_preserve, source_frameworks=["torch"])

  aliased_root = _parse_import("import torch as th")
  res_root = rewriter_preserve.rewrite_import(aliased_root, aliased_root)
  assert isinstance(res_root, cst.Import)
  assert res_root.names[0].asname is not None
  assert isinstance(res_root.names[0].asname.name, cst.Name)
  assert res_root.names[0].asname.name.value == "th"
  assert "th" in rewriter_preserve.state.defined_names

  aliased_sub = _parse_import("import torch.sub as my_sub")
  res_sub = rewriter_preserve.rewrite_import(aliased_sub, aliased_sub)
  assert isinstance(res_sub, cst.Import)
  assert res_sub.names[0].asname is not None
  assert isinstance(res_sub.names[0].asname.name, cst.Name)
  assert res_sub.names[0].asname.name.value == "my_sub"

  plan_override = ResolutionPlan(mappings={"torch": ImportReq(module="jax", alias="j")})
  rewriter_override = _make_rewriter(plan=plan_override, source_frameworks=["torch"])
  res_override = rewriter_override.rewrite_import(aliased_root, aliased_root)
  assert isinstance(res_override, cst.Import)
  assert res_override.names[0].asname is not None
  assert isinstance(res_override.names[0].asname.name, cst.Name)
  assert res_override.names[0].asname.name.value == "j"


def test_rewrite_import_dce_satisfaction_and_commas() -> None:
  """Verify ``rewrite_import`` prunes unused imports, tracks satisfied requirements, and strips trailing commas."""
  rewriter_prune = _make_rewriter(source_frameworks=["torch"], used_names=set())
  torch_imp = _parse_import("import torch")
  assert isinstance(rewriter_prune.rewrite_import(torch_imp, torch_imp), cst.RemovalSentinel)

  torch_unknown = _parse_import("import torch.unknown")
  assert isinstance(rewriter_prune.rewrite_import(torch_unknown, torch_unknown), cst.RemovalSentinel)

  os_imp = _parse_import("import os")
  assert isinstance(rewriter_prune.rewrite_import(os_imp, os_imp), cst.RemovalSentinel)

  rewriter_keep = _make_rewriter(source_frameworks=["torch"], used_names={"torch", "os", "my_sys"})
  res_torch = rewriter_keep.rewrite_import(torch_imp, torch_imp)
  assert isinstance(res_torch, cst.Import)
  assert cst.Module([]).code_for_node(res_torch) == "import torch"

  res_os = rewriter_keep.rewrite_import(os_imp, os_imp)
  assert isinstance(res_os, cst.Import)
  assert isinstance(res_os.names[0].name, cst.Name)
  assert res_os.names[0].name.value == "os"

  aliased_sys = _parse_import("import sys as my_sys")
  res_aliased_sys = rewriter_keep.rewrite_import(aliased_sys, aliased_sys)
  assert isinstance(res_aliased_sys, cst.Import)

  plan_sat = ResolutionPlan(required_imports=[ImportReq("sys"), ImportReq("os", "path", "path")])
  rewriter_sat = _make_rewriter(plan=plan_sat, source_frameworks=["torch"], used_names={"os"})
  sys_imp = _parse_import("import sys")
  rewriter_sat.rewrite_import(sys_imp, sys_imp)
  assert "sys" in rewriter_sat.state.satisfied_injections

  rewriter_sat.rewrite_import(os_imp, os_imp)
  assert "os.path" not in rewriter_sat.state.satisfied_injections

  rewriter_multi = _make_rewriter(
    plan=ResolutionPlan(mappings={"torch": ImportReq("jax", "numpy", "jnp")}),
    source_frameworks={"torch", "torch2"},
    used_names={"sys"},
  )
  multi_imp = _parse_import("import torch, sys, unused_pkg, torch2")
  res_multi = rewriter_multi.rewrite_import(multi_imp, multi_imp)
  assert isinstance(res_multi, cst.Import)
  assert len(res_multi.names) == 2
  assert get_full_name(res_multi.names[0].name) == "jax.numpy"
  assert isinstance(res_multi.names[1].name, cst.Name)
  assert res_multi.names[1].name.value == "sys"
  assert res_multi.names[1].comma == cst.MaybeSentinel.DEFAULT


def test_rewrite_import_from_relative_and_wildcard() -> None:
  """Verify ``rewrite_import_from`` preserves relative imports and non-source wildcards while pruning source wildcards."""
  rewriter = _make_rewriter(source_frameworks=["torch"], used_names={"torch", "x"})

  rel_node = cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("x"))])
  assert rewriter.rewrite_import_from(rel_node, rel_node) is rel_node

  rel_star = cst.ImportFrom(module=None, relative=[cst.Dot()], names=cst.ImportStar())
  assert rewriter.rewrite_import_from(rel_star, rel_star) is rel_star

  src_star = _parse_import_from("from torch import *")
  assert isinstance(rewriter.rewrite_import_from(src_star, src_star), cst.RemovalSentinel)

  other_star = _parse_import_from("from sys import *")
  assert rewriter.rewrite_import_from(other_star, other_star) is other_star


def test_rewrite_import_from_mapped_replacements() -> None:
  """Verify ``rewrite_import_from`` converts single-symbol mapped imports into ``cst.Import`` statements."""
  plan = ResolutionPlan(
    mappings={
      "torch.nn": ImportReq("flax", "nnx", "nnx"),
      "torch.Tensor": ImportReq("jax", "Array"),
      "torch.sin": ImportReq(module="jax.numpy", subcomponent="sin"),
    }
  )
  rewriter = _make_rewriter(plan=plan, source_frameworks=["torch"])

  res_nn = rewriter.rewrite_import_from(
    _parse_import_from("from torch import nn"), _parse_import_from("from torch import nn")
  )
  assert isinstance(res_nn, cst.Import)
  assert get_full_name(res_nn.names[0].name) == "flax.nnx"
  assert res_nn.names[0].asname is not None
  assert isinstance(res_nn.names[0].asname.name, cst.Name)
  assert res_nn.names[0].asname.name.value == "nnx"
  assert "flax.nnx" in rewriter.state.satisfied_injections

  orig_tensor = _parse_import_from("from torch import Tensor")
  res_tensor = rewriter.rewrite_import_from(orig_tensor, orig_tensor)
  assert isinstance(res_tensor, cst.Import)
  assert get_full_name(res_tensor.names[0].name) == "jax.Array"
  assert "jax.Array" in rewriter.state.satisfied_injections

  orig_sin = _parse_import_from("from torch import sin")
  res_sin = rewriter.rewrite_import_from(orig_sin, orig_sin)
  assert isinstance(res_sin, cst.Import)
  assert get_full_name(res_sin.names[0].name) == "jax.numpy.sin"

  plan_root_alias = ResolutionPlan(mappings={"torch.nn": ImportReq(module="jax", subcomponent="", alias="jnn")})
  rewriter_root_alias = _make_rewriter(plan=plan_root_alias, source_frameworks=["torch"])
  orig_nn = _parse_import_from("from torch import nn")
  res_root_alias = rewriter_root_alias.rewrite_import_from(orig_nn, orig_nn)
  assert isinstance(res_root_alias, cst.Import)
  assert get_full_name(res_root_alias.names[0].name) == "jax"
  assert res_root_alias.names[0].asname is not None
  assert isinstance(res_root_alias.names[0].asname.name, cst.Name)
  assert res_root_alias.names[0].asname.name.value == "jnn"
  assert "jax : jnn" in rewriter_root_alias.state.satisfied_injections

  plan_root_plain = ResolutionPlan(mappings={"torch.nn": ImportReq(module="flax", subcomponent=None, alias="flax")})
  rewriter_root_plain = _make_rewriter(plan=plan_root_plain, source_frameworks=["torch"])
  res_root_plain = rewriter_root_plain.rewrite_import_from(orig_nn, orig_nn)
  assert isinstance(res_root_plain, cst.Import)
  assert get_full_name(res_root_plain.names[0].name) == "flax"
  assert res_root_plain.names[0].asname is None


def test_rewrite_import_from_dce_commas_and_parentheses() -> None:
  """Verify ``rewrite_import_from`` performs per-alias DCE and handles trailing commas with/without parentheses."""
  plan = ResolutionPlan()

  rewriter_unused = _make_rewriter(plan=plan, source_frameworks=["torch"], used_names=set())
  assert isinstance(
    rewriter_unused.rewrite_import_from(
      _parse_import_from("from torch import optim"),
      _parse_import_from("from torch import optim"),
    ),
    cst.RemovalSentinel,
  )
  assert isinstance(
    rewriter_unused.rewrite_import_from(
      _parse_import_from("from torch import a, b"),
      _parse_import_from("from torch import a, b"),
    ),
    cst.RemovalSentinel,
  )
  assert isinstance(
    rewriter_unused.rewrite_import_from(
      _parse_import_from("from sys import path"),
      _parse_import_from("from sys import path"),
    ),
    cst.RemovalSentinel,
  )

  rewriter_used = _make_rewriter(plan=plan, source_frameworks=["torch"], used_names={"optim", "h", "sin"})
  orig_optim = _parse_import_from("from torch import optim")
  res_optim = rewriter_used.rewrite_import_from(orig_optim, orig_optim)
  assert isinstance(res_optim, cst.ImportFrom)

  orig_asname = _parse_import_from("from my_mod import helper as h")
  res_asname = rewriter_used.rewrite_import_from(orig_asname, orig_asname)
  assert isinstance(res_asname, cst.ImportFrom)

  orig_non_name = cst.ImportFrom(
    module=cst.Name("my_mod"),
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))],
  )
  assert isinstance(rewriter_used.rewrite_import_from(orig_non_name, orig_non_name), cst.RemovalSentinel)

  orig_from_comma = cst.ImportFrom(
    module=cst.Name("math"),
    names=[
      cst.ImportAlias(name=cst.Name("sin"), comma=cst.Comma()),
      cst.ImportAlias(name=cst.Name("unused")),
    ],
  )
  res_from_comma = rewriter_used.rewrite_import_from(orig_from_comma, orig_from_comma)
  assert isinstance(res_from_comma, cst.ImportFrom)
  assert not isinstance(res_from_comma.names, cst.ImportStar)
  assert res_from_comma.names[0].comma == cst.MaybeSentinel.DEFAULT

  orig_from_rpar = cst.ImportFrom(
    module=cst.Name("math"),
    names=[cst.ImportAlias(name=cst.Name("sin"), comma=cst.Comma())],
    lpar=cst.LeftParen(),
    rpar=cst.RightParen(),
  )
  res_from_rpar = rewriter_used.rewrite_import_from(orig_from_rpar, orig_from_rpar)
  assert isinstance(res_from_rpar, cst.ImportFrom)
  assert not isinstance(res_from_rpar.names, cst.ImportStar)
  assert isinstance(res_from_rpar.names[0].comma, cst.Comma)
