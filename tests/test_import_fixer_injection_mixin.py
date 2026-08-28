"""Test module."""

import libcst as cst
from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq
from typing import Set, List


class MockFixer(InjectionMixin):
  """Test element."""

  def __init__(self, plan: ResolutionPlan) -> None:
    """Test element."""
    self.plan: ResolutionPlan = plan
    self._satisfied_injections: Set[str] = set()
    self._defined_names: Set[str] = set()


def test_leave_module_no_injections() -> None:
  """Test element."""
  plan: ResolutionPlan = ResolutionPlan()
  fixer: MockFixer = MockFixer(plan)
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1


def test_leave_module_with_injection() -> None:
  """Test element."""
  plan: ResolutionPlan = ResolutionPlan(
    required_imports=[ImportReq("torch"), ImportReq("jax", "numpy", "jnp"), ImportReq("torch", "nn", "nn")]
  )
  fixer: MockFixer = MockFixer(plan)
  orig_module: cst.Module = cst.parse_module('"""Doc"""\n\nx = 1')
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)

  assert len(updated.body) == 5  # docstring, torch, jnp, nn, x = 1
  code: str = updated.code
  assert "import torch" in code
  assert "import jax.numpy as jnp" in code
  assert "import torch.nn as nn" in code
  assert '"""Doc"""' in code
  assert "x = 1" in code

  # check index logic (it inserts after docstring)
  assert isinstance(updated.body[0], cst.SimpleStatementLine)
  assert isinstance(updated.body[0].body[0], cst.Expr)


def test_leave_module_deduplication() -> None:
  """Test element."""
  # Test deduplication step at the end of leave_Module
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer: MockFixer = MockFixer(plan)
  orig_module: cst.Module = cst.parse_module("import sys\nimport sys\n")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)

  # Original has 2 import sys, plan has 1 sys.
  # The first "sys" requirement might be checked against satisfied? No, in this test _satisfied_injections is empty,
  # so it will inject a third `import sys`.
  # Then deduplication kicks in and drops all but the first unique one.
  assert len(updated.body) == 1
  assert "import sys" in updated.code


def test_leave_module_already_satisfied() -> None:
  """Test element."""
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer: MockFixer = MockFixer(plan)
  fixer._satisfied_injections.add("sys")
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1
  assert "import sys" not in updated.code


def test_leave_module_already_defined() -> None:
  """Test element."""
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer: MockFixer = MockFixer(plan)
  fixer._defined_names.add("sys")
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1
  assert "import sys" not in updated.code


def test_append_injection() -> None:
  """Test element."""
  fixer: MockFixer = MockFixer(ResolutionPlan())
  injections: List[cst.BaseStatement] = []
  fixer._append_injection(injections, cst.Pass())
  assert len(injections) == 1
