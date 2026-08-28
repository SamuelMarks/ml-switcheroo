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
    self.preserve_source = False
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
  assert getattr(res, "names")[0].name.value == "os"


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
