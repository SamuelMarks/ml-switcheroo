"""
Tests for PaxML Pattern Plugins.

Verifies:
1.  `__init__` -> `setup` renaming.
2.  `super().__init__()` stripping.
3.  Target guard (ignores non-paxml targets).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.pax_patterns import migrate_init_to_setup


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Executes helper."""
  tree = cst.parse_module(code)
  # Note: Hooks work on Call nodes typically, but this plugin targets FunctionDef.
  # This requires the Rewriter to support applying hooks to arbitrary node types if configured.
  # Currently, `PivotRewriter` primarily applies hooks in `leave_Call`.
  # To test this plugin mechanism realistically, we must assume a mechanism exists to
  # register FunctionDef hooks OR we manually invoke it here for unit testing.
  # Given `StructureMixin` handles FunctionDef, we would integrate there.
  # For now, we Unit Test the hook logic directly.
  pass


def test_hook_logic_direct_invocation():
  """
  Unit test the `migrate_init_to_setup` function directly.
  """
  code = """ 
def __init__(self): 
    super().__init__() 
    self.layer = 1
"""
  module = cst.parse_module(code)
  func_def = module.body[0]

  # Setup Context
  ctx = MagicMock()
  ctx.target_fw = "paxml"

  # Run Transformation
  new_node = migrate_init_to_setup(func_def, ctx)

  # Verify Rename
  assert new_node.name.value == "setup"

  # Verify Super Strip
  # New body should only contain `self.layer = 1`
  generated = cst.Module(body=[new_node]).code
  assert "super().__init__()" not in generated
  assert "self.layer = 1" in generated


def test_hook_ignored_for_non_paxml():
  """
  Verify hook does nothing if target is not paxml.
  """
  code = "def __init__(self): pass"
  module = cst.parse_module(code)
  func_def = module.body[0]

  ctx = MagicMock()
  ctx.target_fw = "torch"

  new_node = migrate_init_to_setup(func_def, ctx)

  # Should be unchanged
  assert new_node == func_def
  assert new_node.name.value == "__init__"


def test_hook_ignored_for_non_init():
  """
  Verify hook does nothing for normal methods.
  """
  code = "def forward(self): pass"
  module = cst.parse_module(code)
  func_def = module.body[0]

  ctx = MagicMock()
  ctx.target_fw = "paxml"

  new_node = migrate_init_to_setup(func_def, ctx)

  assert new_node.name.value == "forward"
