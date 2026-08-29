"""Test suite for the Injection Gap module."""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin


class DummyPlan:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the DummyPlan instance."""
    self.required_imports: list[typing.Any] = []


class DummyReq:
  """Docstring."""

  def __init__(
    self, module: str, subcomponent: typing.Optional[str], alias: typing.Optional[str], signature: str
  ) -> None:
    """Initializes the DummyReq instance."""
    self.module = module
    self.subcomponent = subcomponent
    self.alias = alias
    self.signature = signature


class DummyFixer(InjectionMixin, cst.CSTTransformer):
  """Docstring."""

  def __init__(self, plan: DummyPlan) -> None:
    """Initializes the DummyFixer instance."""
    self.plan = plan
    self._satisfied_injections: set[str] = set()
    self._defined_names: set[str] = {"foo"}


def test_injection_skip_defined() -> None:
  """Verifies the behavior of injection skip defined."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="foo", subcomponent=None, alias="foo", signature="import foo"))
  fixer = DummyFixer(plan)
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))  # type: ignore
  assert len(stmts.body) == 0


def test_injection_skip_satisfied() -> None:
  """Verifies the behavior of injection skip satisfied."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="foo", subcomponent=None, alias="foo", signature="import foo"))
  fixer = DummyFixer(plan)
  fixer._satisfied_injections.add("import foo")
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))  # type: ignore
  assert len(stmts.body) == 0


def test_injection_add_imports() -> None:
  """Verifies the behavior of injection add imports."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))
  plan.required_imports.append(
    DummyReq(module="os", subcomponent="path", alias="path", signature="import os.path as path")
  )
  plan.required_imports.append(DummyReq(module="typing", subcomponent=None, alias="t", signature="import typing as t"))
  fixer = DummyFixer(plan)
  stmts: cst.Module = fixer.leave_Module(cst.Module([]), cst.Module([]))  # type: ignore
  assert len(stmts.body) == 3


def test_injection_dedup_and_docstring() -> None:
  """Docstring."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))
  fixer = DummyFixer(plan)
  code: str = '"""doc"""\nfrom __future__ import print_function\nimport sys\nx = 1\n'
  mod: cst.Module = cst.parse_module(code)
  stmts: cst.Module = fixer.leave_Module(mod, mod)  # type: ignore
  assert len(stmts.body) == 4
