"""Test module."""

from typing import List, Set, Union

import libcst as cst

from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


class MockFixer(InjectionMixin):
  """Docstring."""

  def __init__(self, plan: ResolutionPlan) -> None:
    """Docstring."""
    self.plan: ResolutionPlan = plan
    self._satisfied_injections: Set[str] = set()
    self._defined_names: Set[str] = set()


def test_leave_module_no_injections() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan()
  fixer: MockFixer = MockFixer(plan)
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1


def test_leave_module_with_injection() -> None:
  """Docstring."""
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


def test_injection_docstring_and_future() -> None:
  """Test injection placement after docstrings and future imports."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(
      required_imports=[ImportReq("os", None, "os")],
      mappings={},
    ),
  )
  orig: cst.Module = cst.parse_module('"""My Module."""\nfrom __future__ import annotations\n# Some comment\nx = 1')
  res: cst.Module = fixer.leave_Module(orig, orig)
  new_code = res.code

  # Ensure it's inserted after the future import
  assert new_code.index("import os") > new_code.index("__future__")
  assert new_code.index("import os") < new_code.index("x = 1")


def test_injection_deduplication() -> None:
  """Test that identical imports are deduplicated (104->113)."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(required_imports=[], mappings={}),
  )
  orig: cst.Module = cst.parse_module("import os\nimport os\nimport sys\nimport sys\nx = 1\nx = 1")
  res: cst.Module = fixer.leave_Module(orig, orig)
  new_code = res.code
  assert new_code.count("import os") == 1
  assert new_code.count("import sys") == 1
  assert new_code.count("x = 1") == 2


def test_injection_alias_rules() -> None:
  """Test alias rules in injection (65->69)."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(
      required_imports=[
        ImportReq("numpy", None, "numpy"),  # no alias
        ImportReq("math", "m", "m"),  # different alias
      ],
      mappings={},
    ),
  )
  orig: cst.Module = cst.parse_module("x = 1")
  res: cst.Module = fixer.leave_Module(orig, orig)
  new_code = res.code
  assert "import numpy\n" in new_code
  assert "import math.m as m\n" in new_code


def test_injection_no_body() -> None:
  """Test injection placement with empty body (88->94)."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(
      required_imports=[ImportReq("os", None, "os")],
      mappings={},
    ),
  )
  orig: cst.Module = cst.parse_module("")
  res: cst.Module = fixer.leave_Module(orig, orig)
  new_code = res.code
  assert "import os" in new_code


def test_injection_deduplication_other_stmts() -> None:
  """Test deduplication ignoring non-imports (104->113)."""
  fixer: MockFixer = MockFixer(
    plan=ResolutionPlan(required_imports=[], mappings={}),
  )
  orig: cst.Module = cst.parse_module("import os\ndef x(): pass\nimport os\nclass y: pass")
  res: cst.Module = fixer.leave_Module(orig, orig)
  new_code = res.code
  assert new_code.count("import os") == 1
  assert "def x():" in new_code

  # Test deduplication step at the end of leave_Module
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer2: MockFixer = MockFixer(plan)
  orig_module: cst.Module = cst.parse_module("import sys\nimport sys\n")
  updated: cst.Module = fixer2.leave_Module(orig_module, orig_module)

  # Original has 2 import sys, plan has 1 sys.
  # The first "sys" requirement might be checked against satisfied? No, in this test _satisfied_injections is empty,
  # so it will inject a third `import sys`.
  # Then deduplication kicks in and drops all but the first unique one.
  assert len(updated.body) == 1
  assert "import sys" in updated.code


def test_leave_module_already_satisfied() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer: MockFixer = MockFixer(plan)
  fixer._satisfied_injections.add("sys")
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1
  assert "import sys" not in updated.code


def test_leave_module_already_defined() -> None:
  """Docstring."""
  plan: ResolutionPlan = ResolutionPlan(required_imports=[ImportReq("sys")])
  fixer: MockFixer = MockFixer(plan)
  fixer._defined_names.add("sys")
  orig_module: cst.Module = cst.parse_module("x = 1")
  updated: cst.Module = fixer.leave_Module(orig_module, orig_module)
  assert len(updated.body) == 1
  assert "import sys" not in updated.code


def test_append_injection() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(ResolutionPlan())
  injections: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]] = []
  stmt: cst.SimpleStatementLine = cst.SimpleStatementLine(body=[cst.Pass()])
  fixer._append_injection(injections, stmt)
  assert len(injections) == 1
