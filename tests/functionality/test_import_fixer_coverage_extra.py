"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan
from tests.functionality.test_import_fixer_coverage import DummyFixer


def get_full_name_local(node):
  """Docstring."""
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    return get_full_name_local(node.value) + "." + node.attr.value
  return ""


def test_imports_mixin_alias_logic():
  """Docstring."""
  req1 = ImportReq(module="jax", alias="")
  fixer = DummyFixer(ResolutionPlan([req1], {"torch": req1}, {}))
  alias = fixer._make_alias_node(req1)
  assert alias.asname is None

  req2 = ImportReq(module="jax", alias="jax")
  fixer2 = DummyFixer(ResolutionPlan([req2], {"torch": req2}, {}))
  alias2 = fixer2._make_alias_node(req2)
  assert alias2.asname is None

  # 59 -> 60
  req4 = ImportReq(module="jax", alias="j")
  fixer4 = DummyFixer(ResolutionPlan([req4], {"torch": req4}, {}))
  alias4 = fixer4._make_alias_node(req4)
  assert alias4.asname is not None

  req3 = ImportReq(module="jax.numpy", alias="numpy")
  fixer3 = DummyFixer(ResolutionPlan([req3], {"torch": req3}, {}))
  alias3 = fixer3._make_alias_node(req3)
  assert alias3.asname is not None
  assert alias3.asname.name.value == "numpy"


def test_leave_import_branches():
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))
  import_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res = fixer.leave_Import(import_node, import_node)
  assert isinstance(res, cst.RemovalSentinel)

  fixer.preserve_source = True
  res2 = fixer.leave_Import(import_node, import_node)
  assert not isinstance(res2, cst.RemovalSentinel)

  req = ImportReq(module="jax")
  fixer_rep = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer_rep.preserve_source = True
  import_node_rep = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res_rep = fixer_rep.leave_Import(import_node_rep, import_node_rep)
  assert len(res_rep.names) == 1

  fixer.preserve_source = False
  import_node2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  res3 = fixer.leave_Import(import_node2, import_node2)
  assert not isinstance(res3, cst.RemovalSentinel)

  req = ImportReq(module="jax")
  fixer4 = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  import_node3 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("t")))])
  res4 = fixer4.leave_Import(import_node3, import_node3)
  assert res4.names[0].asname is not None
  assert res4.names[0].asname.name.value == "t"

  req2 = ImportReq(module="jax", alias="j")
  fixer5 = DummyFixer(ResolutionPlan([], {"torch": req2}, {}))
  res5 = fixer5.leave_Import(import_node3, import_node3)
  assert res5.names[0].asname.name.value == "j"

  req3 = ImportReq(module="os")
  fixer6 = DummyFixer(ResolutionPlan([req3], {}, {}))
  import_node4 = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  fixer6.leave_Import(import_node4, import_node4)
  assert req3.signature in fixer6._satisfied_injections

  req4 = ImportReq(module="sys")
  fixer7 = DummyFixer(ResolutionPlan([req4], {}, {}))
  fixer7.leave_Import(import_node4, import_node4)
  assert req4.signature not in fixer7._satisfied_injections


def test_leave_import_from_branches():
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))

  import_from_none = cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("a"))])
  res1 = fixer.leave_ImportFrom(import_from_none, import_from_none)
  assert res1 == import_from_none

  import_from_star = cst.ImportFrom(module=cst.Name("torch"), names=cst.ImportStar())
  res2 = fixer.leave_ImportFrom(import_from_star, import_from_star)
  assert isinstance(res2, cst.RemovalSentinel)

  import_from_star2 = cst.ImportFrom(module=cst.Name("os"), names=cst.ImportStar())
  res3 = fixer.leave_ImportFrom(import_from_star2, import_from_star2)
  assert not isinstance(res3, cst.RemovalSentinel)

  fixer.preserve_source = True
  res_ps = fixer.leave_ImportFrom(import_from_star, import_from_star)
  assert not isinstance(res_ps, cst.RemovalSentinel)
  fixer.preserve_source = False

  req = ImportReq(module="jax.numpy", subcomponent="sin")
  fixer2 = DummyFixer(ResolutionPlan([], {"torch.sin": req}, {}))
  import_from_mapping = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("sin"))])
  res4 = fixer2.leave_ImportFrom(import_from_mapping, import_from_mapping)
  assert isinstance(res4, cst.Import)
  assert get_full_name_local(res4.names[0].name) == "jax.numpy.sin"

  req_no_sub = ImportReq(module="jax")
  fixer3 = DummyFixer(ResolutionPlan([], {"torch.nn": req_no_sub}, {}))
  import_from_mapping2 = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res5 = fixer3.leave_ImportFrom(import_from_mapping2, import_from_mapping2)
  assert isinstance(res5, cst.Import)
  assert get_full_name_local(res5.names[0].name) == "jax"

  import_from_multi = cst.ImportFrom(
    module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("sin")), cst.ImportAlias(name=cst.Name("cos"))]
  )
  res6 = fixer.leave_ImportFrom(import_from_multi, import_from_multi)
  assert isinstance(res6, cst.RemovalSentinel)

  import_from_unmapped = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("unknown"))])
  res7 = fixer.leave_ImportFrom(import_from_unmapped, import_from_unmapped)
  assert isinstance(res7, cst.RemovalSentinel)

  import_from_os = cst.ImportFrom(module=cst.Name("os"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res8 = fixer.leave_ImportFrom(import_from_os, import_from_os)
  assert not isinstance(res8, cst.RemovalSentinel)

  fixer.preserve_source = True
  res9 = fixer.leave_ImportFrom(import_from_unmapped, import_from_unmapped)
  assert not isinstance(res9, cst.RemovalSentinel)


def test_imports_mixin_119_121():
  """Docstring."""
  req = ImportReq(module="jax")
  fixer = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer.preserve_source = True
  import_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("torch2"))])
  fixer.source_fws = {"torch", "torch2"}
  fixer.leave_Import(import_node, import_node)


def test_imports_mixin_119_exhaustive():
  """Docstring."""
  # 1. preserve=False
  fixer1 = DummyFixer(ResolutionPlan([], {}, {}))
  fixer1.preserve_source = False
  fixer1.source_fws = {"torch"}
  node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  fixer1.leave_Import(node, node)

  # 2. preserve=True, repl=False
  fixer2 = DummyFixer(ResolutionPlan([], {}, {}))
  fixer2.preserve_source = True
  fixer2.source_fws = {"torch"}
  fixer2.leave_Import(node, node)

  # 3. preserve=True, repl=True (from earlier in loop)
  req = ImportReq(module="jax")
  fixer3 = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer3.preserve_source = True
  fixer3.source_fws = {"torch", "torch2"}
  node2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("torch2"))])
  fixer3.leave_Import(node2, node2)

  # 4. preserve=False, repl=True
  fixer4 = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer4.preserve_source = False
  fixer4.source_fws = {"torch", "torch2"}
  fixer4.leave_Import(node2, node2)
