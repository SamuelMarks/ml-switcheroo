"""Module docstring."""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeMixin
from ml_switcheroo.core.import_fixer.base import BaseImportFixer
from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


class DummyFixer(AttributeMixin, ImportMixin, InjectionMixin, BaseImportFixer):
  """Docstring."""

  def __init__(self, plan: ResolutionPlan) -> None:
    """Docstring."""
    self.plan = plan
    self.source_fws: set[str] = {"torch"}
    self.target_fw = "jax"
    self.used_names = set()
    self._defined_names: set[str] = set()
    self._path_to_alias: dict[str, str] = {}
    self._satisfied_injections: set[str] = set()
    self.target_module = "jax"


def test_attributes_mixin_missing_branches() -> None:
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))
  if hasattr(fixer, "_defined_names"):
    delattr(fixer, "_defined_names")
  if hasattr(fixer, "_path_to_alias"):
    delattr(fixer, "_path_to_alias")
  if hasattr(fixer, "target_fw"):
    delattr(fixer, "target_fw")

  attr_node = cst.Attribute(value=cst.Name("something"), attr=cst.Name("nn"))
  node = cst.Attribute(value=attr_node, attr=cst.Name("func"))
  res: typing.Any = fixer.leave_Attribute(node, node)
  assert res == node


def test_imports_mixin_not_in_source_fws() -> None:
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))
  import_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  res: typing.Any = fixer.leave_Import(import_node, import_node)
  assert isinstance(res, cst.RemovalSentinel)


def test_injection_mixin_no_alias() -> None:
  """Docstring."""
  req = ImportReq(module="os", subcomponent="", alias="")
  plan = ResolutionPlan(required_imports=[req], mappings={}, path_to_alias={})
  fixer = DummyFixer(plan)
  mod = cst.Module(body=[])
  res: typing.Any = fixer.leave_Module(mod, mod)  # type: ignore
  assert len(res.body) == 1


def test_injection_mixin_alias_dot() -> None:
  """Docstring."""
  req = ImportReq(module="os.path", subcomponent="", alias="")
  plan = ResolutionPlan(required_imports=[req], mappings={}, path_to_alias={})
  fixer = DummyFixer(plan)
  mod = cst.Module(body=[])
  res: typing.Any = fixer.leave_Module(mod, mod)  # type: ignore
  assert len(res.body) == 1


# --- Merged from test_import_fixer_coverage_extra.py ---


def get_full_name_local(node: typing.Any) -> str:
  """Docstring."""
  if isinstance(node, cst.Name):
    return node.value
  elif isinstance(node, cst.Attribute):
    return get_full_name_local(node.value) + "." + node.attr.value
  return ""


def test_imports_mixin_alias_logic() -> None:
  """Docstring."""
  req1 = ImportReq(module="jax", alias="")
  fixer = DummyFixer(ResolutionPlan([req1], {"torch": req1}, {}))
  alias: typing.Any = fixer._make_alias_node(req1)
  assert alias.asname is None

  req2 = ImportReq(module="jax", alias="jax")
  fixer2 = DummyFixer(ResolutionPlan([req2], {"torch": req2}, {}))
  alias2: typing.Any = fixer2._make_alias_node(req2)
  assert alias2.asname is None

  # 59 -> 60
  req4 = ImportReq(module="jax", alias="j")
  fixer4 = DummyFixer(ResolutionPlan([req4], {"torch": req4}, {}))
  alias4: typing.Any = fixer4._make_alias_node(req4)
  assert alias4.asname is not None

  req3 = ImportReq(module="jax.numpy", alias="numpy")
  fixer3 = DummyFixer(ResolutionPlan([req3], {"torch": req3}, {}))
  alias3: typing.Any = fixer3._make_alias_node(req3)
  assert alias3.asname is not None
  assert alias3.asname.name.value == "numpy"


def test_leave_import_branches() -> None:
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))
  import_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res: typing.Any = fixer.leave_Import(import_node, import_node)
  assert isinstance(res, cst.RemovalSentinel)

  fixer.used_names = {
    "torch",
    "optim",
    "nn",
    "sys",
    "re",
    "math",
    "test_pkg",
    "other",
    "some_alias",
    "a",
    "b",
    "c",
    "my_pkg",
    "math_alias",
    "x",
    "y",
  }
  res2: typing.Any = fixer.leave_Import(import_node, import_node)
  assert not isinstance(res2, cst.RemovalSentinel)

  req = ImportReq(module="jax")
  fixer_rep = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer_rep.used_names = {"math_alias"}
  import_node_rep = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  res_rep: typing.Any = fixer_rep.leave_Import(import_node_rep, import_node_rep)
  assert len(res_rep.names) == 1

  fixer.used_names = set()
  import_node2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("os"))])
  res3: typing.Any = fixer.leave_Import(import_node2, import_node2)
  assert isinstance(res3, cst.RemovalSentinel)

  req_j = ImportReq(module="jax")
  fixer4 = DummyFixer(ResolutionPlan([], {"torch": req_j}, {}))
  import_node3 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"), asname=cst.AsName(name=cst.Name("t")))])
  res4: typing.Any = fixer4.leave_Import(import_node3, import_node3)
  assert res4.names[0].asname is not None
  assert res4.names[0].asname.name.value == "t"

  req2 = ImportReq(module="jax", alias="j")
  fixer5 = DummyFixer(ResolutionPlan([], {"torch": req2}, {}))
  res5: typing.Any = fixer5.leave_Import(import_node3, import_node3)
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


def test_leave_import_from_branches() -> None:
  """Docstring."""
  fixer = DummyFixer(ResolutionPlan([], {}, {}))

  import_from_none = cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(name=cst.Name("a"))])
  res1: typing.Any = fixer.leave_ImportFrom(import_from_none, import_from_none)
  assert res1 == import_from_none

  import_from_star = cst.ImportFrom(module=cst.Name("torch"), names=cst.ImportStar())
  res2: typing.Any = fixer.leave_ImportFrom(import_from_star, import_from_star)
  assert isinstance(res2, cst.RemovalSentinel)

  import_from_star2 = cst.ImportFrom(module=cst.Name("os"), names=cst.ImportStar())
  res3: typing.Any = fixer.leave_ImportFrom(import_from_star2, import_from_star2)
  assert not isinstance(res3, cst.RemovalSentinel)

  fixer.used_names = {
    "torch",
    "optim",
    "nn",
    "sys",
    "re",
    "math",
    "test_pkg",
    "other",
    "some_alias",
    "a",
    "b",
    "c",
    "my_pkg",
    "math_alias",
    "x",
    "y",
  }
  res_ps: typing.Any = fixer.leave_ImportFrom(import_from_star, import_from_star)
  assert isinstance(res_ps, cst.RemovalSentinel)
  fixer.used_names = set()

  req = ImportReq(module="jax.numpy", subcomponent="sin")
  fixer2 = DummyFixer(ResolutionPlan([], {"torch.sin": req}, {}))
  import_from_mapping = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("sin"))])
  res4: typing.Any = fixer2.leave_ImportFrom(import_from_mapping, import_from_mapping)
  assert isinstance(res4, cst.Import)
  assert get_full_name_local(res4.names[0].name) == "jax.numpy.sin"

  req_no_sub = ImportReq(module="jax")
  fixer3 = DummyFixer(ResolutionPlan([], {"torch.nn": req_no_sub}, {}))
  import_from_mapping2 = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])
  res5: typing.Any = fixer3.leave_ImportFrom(import_from_mapping2, import_from_mapping2)
  assert isinstance(res5, cst.Import)
  assert get_full_name_local(res5.names[0].name) == "jax"

  import_from_multi = cst.ImportFrom(
    module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("sin")), cst.ImportAlias(name=cst.Name("cos"))]
  )
  res6: typing.Any = fixer.leave_ImportFrom(import_from_multi, import_from_multi)
  assert isinstance(res6, cst.RemovalSentinel)

  import_from_unmapped = cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("unknown"))])
  res7: typing.Any = fixer.leave_ImportFrom(import_from_unmapped, import_from_unmapped)
  assert isinstance(res7, cst.RemovalSentinel)

  import_from_os = cst.ImportFrom(module=cst.Name("os"), names=[cst.ImportAlias(name=cst.Name("path"))])
  res8: typing.Any = fixer.leave_ImportFrom(import_from_os, import_from_os)
  assert isinstance(res8, cst.RemovalSentinel)

  fixer.used_names = {
    "torch",
    "optim",
    "nn",
    "sys",
    "re",
    "math",
    "test_pkg",
    "other",
    "some_alias",
    "a",
    "b",
    "c",
    "my_pkg",
    "math_alias",
    "x",
    "y",
    "unknown",
  }
  res9: typing.Any = fixer.leave_ImportFrom(import_from_unmapped, import_from_unmapped)
  assert not isinstance(res9, cst.RemovalSentinel)


def test_imports_mixin_119_121() -> None:
  """Docstring."""
  req = ImportReq(module="jax")
  fixer = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer.used_names = {
    "torch",
    "optim",
    "nn",
    "sys",
    "re",
    "math",
    "test_pkg",
    "other",
    "some_alias",
    "a",
    "b",
    "c",
    "my_pkg",
    "math_alias",
    "x",
    "y",
  }
  import_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("torch2"))])
  fixer.source_fws = {"torch", "torch2"}
  fixer.leave_Import(import_node, import_node)


def test_imports_mixin_119_exhaustive() -> None:
  """Docstring."""
  # 1. preserve=False
  fixer1 = DummyFixer(ResolutionPlan([], {}, {}))
  fixer1.used_names = set()
  fixer1.source_fws = {"torch"}
  node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  fixer1.leave_Import(node, node)

  # 2. preserve=True, repl=False
  fixer2 = DummyFixer(ResolutionPlan([], {}, {}))
  fixer2.used_names = {"my_pkg", "other", "a", "b", "c", "x", "y"}
  fixer2.source_fws = {"torch"}
  fixer2.leave_Import(node, node)

  # 3. preserve=True, repl=True (from earlier in loop)
  req = ImportReq(module="jax")
  fixer3 = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer3.used_names = {"my_pkg", "other", "a", "b", "c", "x", "y"}
  fixer3.source_fws = {"torch", "torch2"}
  node2 = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch")), cst.ImportAlias(name=cst.Name("torch2"))])
  fixer3.leave_Import(node2, node2)

  # 4. preserve=False, repl=True
  fixer4 = DummyFixer(ResolutionPlan([], {"torch": req}, {}))
  fixer4.used_names = set()
  fixer4.source_fws = {"torch", "torch2"}
  fixer4.leave_Import(node2, node2)


# --- Merged from test_import_fixer_coverage_extra_injection.py ---


def test_injection_mixin_branches() -> None:
  """Docstring."""
  # 48 -> 49: check_name in _defined_names
  req1 = ImportReq(module="jax")
  fixer = DummyFixer(ResolutionPlan([req1], {}, {}))
  fixer._defined_names = {"jax"}
  mod1 = cst.parse_module("")
  res1: typing.Any = fixer.leave_Module(mod1, mod1)
  assert len(res1.body) == 0

  # 63 -> 65: req.alias == leaf
  # 65 -> 67: "." in nm
  req2 = ImportReq(module="jax.numpy", alias="numpy")
  fixer2 = DummyFixer(ResolutionPlan([req2], {}, {}))
  fixer2._defined_names = set()
  mod2 = cst.parse_module("")
  res2: typing.Any = fixer2.leave_Module(mod2, mod2)
  assert len(res2.body) == 1
  # asname should be 'numpy'

  # 65 -> 69: "." not in nm
  req3 = ImportReq(module="jax", alias="jax")
  fixer3 = DummyFixer(ResolutionPlan([req3], {}, {}))
  fixer3._defined_names = set()
  mod3 = cst.parse_module("")
  res3: typing.Any = fixer3.leave_Module(mod3, mod3)
  assert len(res3.body) == 1
  # asname should be None

  # 89 -> 90: is_docstring
  # 104 -> 113: not SimpleStatementLine (e.g., If or FunctionDef)
  # 114 -> 115: sig in seen_imports (duplicate imports)

  code: str = '"""Docstring"""\nimport jax\ndef foo(): pass\nimport jax\n'
  mod = cst.parse_module(code)
  fixer4 = DummyFixer(ResolutionPlan([], {}, {}))
  mod_out: typing.Any = fixer4.leave_Module(mod, mod)

  assert len(mod_out.body) == 3
