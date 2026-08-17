"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan
from tests.functionality.test_import_fixer_coverage import DummyFixer


def test_injection_mixin_branches():
  """Docstring."""
  # 48 -> 49: check_name in _defined_names
  req1 = ImportReq(module="jax")
  fixer = DummyFixer(ResolutionPlan([req1], {}, {}))
  fixer._defined_names = {"jax"}
  mod1 = cst.parse_module("")
  res1 = fixer.leave_Module(mod1, mod1)
  assert len(res1.body) == 0

  # 63 -> 65: req.alias == leaf
  # 65 -> 67: "." in nm
  req2 = ImportReq(module="jax.numpy", alias="numpy")
  fixer2 = DummyFixer(ResolutionPlan([req2], {}, {}))
  fixer2._defined_names = set()
  mod2 = cst.parse_module("")
  res2 = fixer2.leave_Module(mod2, mod2)
  assert len(res2.body) == 1
  # asname should be 'numpy'

  # 65 -> 69: "." not in nm
  req3 = ImportReq(module="jax", alias="jax")
  fixer3 = DummyFixer(ResolutionPlan([req3], {}, {}))
  fixer3._defined_names = set()
  mod3 = cst.parse_module("")
  res3 = fixer3.leave_Module(mod3, mod3)
  assert len(res3.body) == 1
  # asname should be None

  # 89 -> 90: is_docstring
  # 104 -> 113: not SimpleStatementLine (e.g., If or FunctionDef)
  # 114 -> 115: sig in seen_imports (duplicate imports)

  code = '''"""Docstring"""
import jax
def foo(): pass
import jax
'''
  mod = cst.parse_module(code)
  fixer4 = DummyFixer(ResolutionPlan([], {}, {}))
  mod_out = fixer4.leave_Module(mod, mod)

  assert len(mod_out.body) == 3
