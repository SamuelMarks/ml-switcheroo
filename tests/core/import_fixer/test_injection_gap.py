"""Auto-generated doc."""

import libcst as cst
from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin


class DummyPlan:
  """Auto-generated doc."""

  def __init__(self):
    """Auto-generated doc."""
    self.required_imports = []


class DummyReq:
  """Auto-generated doc."""

  def __init__(self, module, subcomponent, alias, signature):
    """Auto-generated doc."""
    self.module = module
    self.subcomponent = subcomponent
    self.alias = alias
    self.signature = signature


class DummyFixer(InjectionMixin, cst.CSTTransformer):
  """Auto-generated doc."""

  def __init__(self, plan):
    """Auto-generated doc."""
    self.plan = plan
    self._satisfied_injections = set()
    self._defined_names = {"foo"}


def test_injection_skip_defined():
  """Auto-generated doc."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="foo", subcomponent=None, alias="foo", signature="import foo"))

  fixer = DummyFixer(plan)
  stmts = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 0


def test_injection_skip_satisfied():
  """Auto-generated doc."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="foo", subcomponent=None, alias="foo", signature="import foo"))

  fixer = DummyFixer(plan)
  fixer._satisfied_injections.add("import foo")
  stmts = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 0


def test_injection_add_imports():
  """Auto-generated doc."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))
  plan.required_imports.append(
    DummyReq(module="os", subcomponent="path", alias="path", signature="import os.path as path")
  )
  plan.required_imports.append(DummyReq(module="typing", subcomponent=None, alias="t", signature="import typing as t"))

  fixer = DummyFixer(plan)
  stmts = fixer.leave_Module(cst.Module([]), cst.Module([]))
  assert len(stmts.body) == 3


def test_injection_dedup_and_docstring():
  """Auto-generated doc."""
  plan = DummyPlan()
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))
  plan.required_imports.append(DummyReq(module="sys", subcomponent=None, alias=None, signature="import sys"))  # Dupe

  fixer = DummyFixer(plan)
  code = '"""doc"""\nfrom __future__ import print_function\nimport sys\nx = 1\n'
  mod = cst.parse_module(code)

  stmts = fixer.leave_Module(mod, mod)
  assert len(stmts.body) == 4  # doc, future, import sys, x=1
