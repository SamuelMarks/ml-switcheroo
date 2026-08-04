"""Unit tests for the InjectionMixin class.

This module contains tests to verify that the `InjectionMixin` class correctly
post-processes LibCST Module ASTs to inject required imports according to a
resolution plan. It verifies handling of basic injections, docstring placement,
__future__ imports, duplicate imports, and already satisfied/defined names.
"""

import libcst as cst
from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq


class MockFixer(InjectionMixin):
  """A mock import fixer subclassing InjectionMixin for unit testing.

  This class implements the `InjectionMixin` interface to simulate the behavior of
  an AST fixer that has a predetermined set of required imports. It populates
  a mock `ResolutionPlan` with requirements for 'jax.numpy' and 'typing' to
  facilitate tests of the import-injection logic.
  """

  def __init__(self) -> None:
    """Initializes MockFixer with a mock ResolutionPlan.

    The plan is prepopulated with specific required imports (e.g., 'jax.numpy' as
    'jnp' and 'typing') to support testing of various import-injection and
    deduplication behaviors.

    Returns:
        None

    """
    self.plan = ResolutionPlan()
    self._satisfied_injections = set()
    self._defined_names = set()

    # Will result in 'import jax.numpy as jnp'
    self.plan.required_imports.append(ImportReq(module="jax", subcomponent="numpy", alias="jnp"))

    # Will result in 'import typing'
    self.plan.required_imports.append(ImportReq(module="typing"))


def test_injectionmixin_leave_module_basic() -> None:
  """Tests basic import injection into a simple module AST.

  Verifies that if a module does not contain any prior imports, docstrings, or
  __future__ statements, the required imports from the plan are prepended
  cleanly to the beginning of the module body.

  Returns:
      None

  """
  fixer = MockFixer()
  code = """
def test():
    pass
"""
  tree = cst.parse_module(code)
  result = fixer.leave_Module(tree, tree)

  code_result = result.code
  assert "import jax.numpy as jnp" in code_result
  assert "import typing" in code_result
  assert "def test():" in code_result


def test_injectionmixin_leave_module_docstring() -> None:
  """Tests that required imports are injected *after* an existing module-level docstring.

  Verifies that when a module-level docstring is present at the beginning of the file,
  injected imports are inserted immediately following the docstring rather than before
  it, preserving valid Python syntax and structure.

  Returns:
      None

  """
  fixer = MockFixer()
  code = '"""Module docstring"""\ndef test(): pass'
  tree = cst.parse_module(code)
  result = fixer.leave_Module(tree, tree)

  code_result = result.code
  # Should be inserted after the docstring
  assert code_result.startswith('"""Module docstring"""')
  assert "import jax.numpy as jnp" in code_result


def test_injectionmixin_leave_module_future_import() -> None:
  """Tests that required imports are injected *after* __future__ imports.

  Verifies that when a module begins with a `from __future__ import ...` statement,
  the injected imports are correctly placed after the __future__ imports, as required
  by Python grammar constraints.

  Returns:
      None

  """
  fixer = MockFixer()
  code = "from __future__ import annotations\ndef test(): pass"
  tree = cst.parse_module(code)
  result = fixer.leave_Module(tree, tree)

  code_result = result.code
  assert code_result.startswith("from __future__ import annotations")
  assert "import jax.numpy as jnp" in code_result


def test_injectionmixin_leave_module_deduplication() -> None:
  """Tests deduplication of injected imports against existing ones.

  Verifies that if identical imports are already present in the target module,
  the injection logic does not insert duplicate copies of those imports.

  Returns:
      None

  """
  fixer = MockFixer()
  code = """
import jax.numpy as jnp
import typing

def test():
    pass
"""
  tree = cst.parse_module(code)
  # the injections should not duplicate the existing imports because
  # the deduplication logic in leave_Module checks for exact signature match.
  # We must mark them as not satisfied to test deduplication.
  result = fixer.leave_Module(tree, tree)

  # Should only appear once
  assert result.code.count("import jax.numpy as jnp") == 1
  assert result.code.count("import typing") == 1


def test_injectionmixin_already_satisfied() -> None:
  """Tests that imports are skipped if already marked as satisfied.

  Verifies that if a required import is present in the `_satisfied_injections` set,
  it is not injected into the final module AST.

  Returns:
      None

  """
  fixer = MockFixer()
  fixer._satisfied_injections.add("jax.numpy : jnp")
  code = "def test(): pass"

  tree = cst.parse_module(code)
  result = fixer.leave_Module(tree, tree)

  assert "import jax.numpy as jnp" not in result.code
  assert "import typing" in result.code


def test_injectionmixin_already_defined() -> None:
  """Tests that imports are skipped if their target names are already defined.

  Verifies that if an alias or target name is present in the `_defined_names` set,
  the fixer does not attempt to inject the associated import into the module AST.

  Returns:
      None

  """
  fixer = MockFixer()
  fixer._defined_names.add("jnp")
  code = "def test(): pass"

  tree = cst.parse_module(code)
  result = fixer.leave_Module(tree, tree)

  assert "import jax.numpy as jnp" not in result.code
  assert "import typing" in result.code
