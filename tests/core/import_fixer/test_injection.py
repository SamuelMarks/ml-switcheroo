"""Unit tests for ``ImportInjector`` in the import fixer module."""

import libcst as cst

from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.injection_mixin import ImportInjector
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


def _make_injector(
  required_imports: list[ImportReq] | None = None,
  defined_names: set[str] | None = None,
  satisfied_injections: set[str] | None = None,
) -> ImportInjector:
  """Construct an ``ImportInjector`` initialized with the given plan and state sets.

  Args:
      required_imports: Optional list of ``ImportReq`` items for ``ResolutionPlan``.
      defined_names: Optional set of already bound local names.
      satisfied_injections: Optional set of already satisfied ``ImportReq.signature`` strings.

  Returns:
      Configured ``ImportInjector`` instance.
  """
  plan = ResolutionPlan(required_imports=required_imports if required_imports is not None else [])
  state = ImportFixerState.create(plan)
  if defined_names is not None:
    state.defined_names.update(defined_names)
  if satisfied_injections is not None:
    state.satisfied_injections.update(satisfied_injections)
  return ImportInjector(state)


def test_inject_module_imports_no_requirements() -> None:
  """Verify ``inject_module_imports`` leaves module body intact when no imports are required."""
  injector = _make_injector()
  orig_module = cst.parse_module("x = 1")
  updated = injector.inject_module_imports(orig_module, orig_module)
  assert len(updated.body) == 1
  assert updated.code == "x = 1"


def test_inject_module_imports_basic_and_empty_body() -> None:
  """Verify ``inject_module_imports`` prepends required imports into regular and empty modules."""
  injector = _make_injector(
    required_imports=[
      ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
      ImportReq(module="typing"),
    ]
  )
  tree = cst.parse_module(
    """
def test():
  pass
"""
  )
  result = injector.inject_module_imports(tree, tree)
  assert "import jax.numpy as jnp" in result.code
  assert "import typing" in result.code
  assert "def test():" in result.code

  empty_injector = _make_injector(
    required_imports=[
      ImportReq(module="os", subcomponent="", alias=""),
      ImportReq(module="sys.path", subcomponent="", alias=""),
    ]
  )
  empty_mod = cst.parse_module("")
  empty_res = empty_injector.inject_module_imports(empty_mod, empty_mod)
  assert len(empty_res.body) == 2
  assert "import os" in empty_res.code
  assert "import sys.path" in empty_res.code


def test_inject_module_imports_placement_after_docstring_and_future() -> None:
  """Verify injected imports appear after module docstrings and ``__future__`` imports."""
  reqs = [
    ImportReq("torch"),
    ImportReq("jax", "numpy", "jnp"),
    ImportReq("torch", "nn", "nn"),
  ]
  injector_doc = _make_injector(required_imports=reqs)
  doc_mod = cst.parse_module(
    '''"""Doc"""

x = 1'''
  )
  doc_res = injector_doc.inject_module_imports(doc_mod, doc_mod)
  assert len(doc_res.body) == 5
  assert doc_res.code.startswith('"""Doc"""')
  assert "import torch" in doc_res.code
  assert "import jax.numpy as jnp" in doc_res.code
  assert "import torch.nn as nn" in doc_res.code

  injector_only_doc = _make_injector(required_imports=[ImportReq("os")])
  only_doc_mod = cst.parse_module(
    '''"""Header only."""
'''
  )
  only_doc_res = injector_only_doc.inject_module_imports(only_doc_mod, only_doc_mod)
  assert only_doc_res.code.startswith('"""Header only."""')
  assert "import os" in only_doc_res.code

  injector_future = _make_injector(required_imports=[ImportReq("jax", "numpy", "jnp")])
  future_mod = cst.parse_module(
    """from __future__ import annotations
def test(): pass"""
  )
  future_res = injector_future.inject_module_imports(future_mod, future_mod)
  assert future_res.code.startswith("from __future__ import annotations")
  assert "import jax.numpy as jnp" in future_res.code

  injector_both = _make_injector(required_imports=[ImportReq("os", None, "os")])
  both_mod = cst.parse_module(
    '''"""My Module."""
from __future__ import annotations
# Some comment
x = 1'''
  )
  both_res = injector_both.inject_module_imports(both_mod, both_mod)
  assert both_res.code.index("import os") > both_res.code.index("__future__")
  assert both_res.code.index("import os") < both_res.code.index("x = 1")


def test_inject_module_imports_alias_rules() -> None:
  """Verify ``inject_module_imports`` formats plain, dotted, redundant-alias, and renamed-alias imports."""
  injector = _make_injector(
    required_imports=[
      ImportReq("numpy", None, "numpy"),
      ImportReq("os", "path", "path"),
      ImportReq("urllib", "request", None),
      ImportReq("typing", None, "t"),
      ImportReq("math", "m", "m"),
    ]
  )
  mod = cst.parse_module("x = 1")
  res = injector.inject_module_imports(mod, mod)
  lines = res.code.splitlines()

  assert "import numpy" in lines
  assert "import os.path as path" in lines
  assert "import urllib.request" in lines
  assert "import typing as t" in lines
  assert "import math.m as m" in lines
  assert "request" in injector.state.defined_names

  dotted_same_leaf = _make_injector(required_imports=[ImportReq("jax.numpy", None, "numpy")])
  dotted_res = dotted_same_leaf.inject_module_imports(mod, mod)
  assert "import jax.numpy as numpy" in dotted_res.code.splitlines()


def test_inject_module_imports_skip_satisfied_and_defined() -> None:
  """Verify ``inject_module_imports`` skips requirements already satisfied or locally defined."""
  req_jnp = ImportReq(module="jax", subcomponent="numpy", alias="jnp")
  req_sys = ImportReq(module="sys")
  req_foo = ImportReq(module="foo", subcomponent=None, alias="foo")

  injector_sat = _make_injector(
    required_imports=[req_jnp, req_sys],
    satisfied_injections={req_jnp.signature},
  )
  mod = cst.parse_module("def test(): pass")
  res_sat = injector_sat.inject_module_imports(mod, mod)
  assert "import jax.numpy as jnp" not in res_sat.code
  assert "import sys" in res_sat.code

  injector_def = _make_injector(
    required_imports=[req_jnp, req_foo, req_sys],
    defined_names={"jnp", "foo"},
  )
  res_def = injector_def.inject_module_imports(mod, mod)
  assert "import jax.numpy as jnp" not in res_def.code
  assert "import foo" not in res_def.code
  assert "import sys" in res_def.code


def test_inject_module_imports_deduplication() -> None:
  """Verify ``inject_module_imports`` deduplicates repeated imports while keeping non-import statements."""
  injector = _make_injector()
  dup_mod = cst.parse_module(
    """import os
import os
import sys
import sys
x = 1
x = 1"""
  )
  dup_res = injector.inject_module_imports(dup_mod, dup_mod)
  assert dup_res.code.count("import os") == 1
  assert dup_res.code.count("import sys") == 1
  assert dup_res.code.count("x = 1") == 2

  injector_existing = _make_injector(
    required_imports=[
      ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
      ImportReq(module="typing"),
      ImportReq(module="typing"),
    ]
  )
  existing_code = '''"""doc"""
from __future__ import annotations
import jax.numpy as jnp
import typing
def f(): pass
import typing
class C: pass
'''
  existing_mod = cst.parse_module(existing_code)
  existing_res = injector_existing.inject_module_imports(existing_mod, existing_mod)
  assert existing_res.code.count("import jax.numpy as jnp") == 1
  assert existing_res.code.count("import typing") == 1
  assert "def f():" in existing_res.code
  assert "class C:" in existing_res.code

  empty_stmt_mod = cst.Module(body=[cst.SimpleStatementLine(body=[])])
  empty_stmt_res = injector.inject_module_imports(empty_stmt_mod, empty_stmt_mod)
  assert len(empty_stmt_res.body) == 1


def test_append_injection() -> None:
  """Verify ``append_injection`` appends a statement node to the pending injection list."""
  injector = _make_injector()
  injections: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
  stmt = cst.SimpleStatementLine(body=[cst.Pass()])
  injector.append_injection(injections, stmt)
  assert injections == [stmt]
