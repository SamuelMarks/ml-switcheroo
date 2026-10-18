"""Test suite for the Injection Gap module."""

import libcst as cst

from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


class DummyFixer(InjectionMixin):
  """Test helper class DummyFixer."""

  def __init__(self, plan: ResolutionPlan) -> None:
    """Initializes the DummyFixer instance.

    Args:
        plan: Test argument or fixture.
    """
    self.plan = plan
    self._satisfied_injections: set[str] = set()
    self._defined_names: set[str] = {"foo"}


def test_injection_skip_defined() -> None:
  """Verifies the behavior of injection skip defined."""
  plan = ResolutionPlan()
  plan.required_imports.append(ImportReq(module="foo", subcomponent=None, alias="foo"))
  fixer = DummyFixer(plan)
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 0


def test_injection_skip_satisfied() -> None:
  """Verifies the behavior of injection skip satisfied."""
  plan = ResolutionPlan()
  req = ImportReq(module="bar", subcomponent=None, alias="bar")
  plan.required_imports.append(req)
  fixer = DummyFixer(plan)
  fixer._satisfied_injections.add(req.signature)
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 0


def test_injection_add_imports() -> None:
  """Verifies the behavior of injection add imports."""
  plan = ResolutionPlan()
  plan.required_imports.append(ImportReq(module="sys", subcomponent=None, alias=None))
  plan.required_imports.append(ImportReq(module="os", subcomponent="path", alias="path"))
  plan.required_imports.append(ImportReq(module="typing", subcomponent=None, alias="t"))
  fixer = DummyFixer(plan)
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 3


def test_injection_dedup_and_docstring() -> None:
  """Verify injection dedup and docstring behavior."""
  plan = ResolutionPlan()
  plan.required_imports.append(ImportReq(module="sys", subcomponent=None, alias=None))
  plan.required_imports.append(ImportReq(module="sys", subcomponent=None, alias=None))
  fixer = DummyFixer(plan)
  code: str = '"""doc"""\nfrom __future__ import print_function\nimport sys\nx = 1\n'
  mod: cst.Module = cst.parse_module(code)
  stmts: cst.Module = fixer.leave_Module(mod, mod)
  assert len(stmts.body) == 4
