"""
Integration Tests for Automatic Plugin Wiring.

Verifies that:
1. Defining a hook with `auto_wire` metadata correctly registers it.
2. The SemanticsManager picks up this metadata without JSON files.
3. The ASTEngine performs translation using the auto-wired definition.
"""

import pytest
import libcst as cst
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.hooks import register_hook, clear_hooks, HookContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture(autouse=True)
def clean_env():
  clear_hooks()
  yield
  clear_hooks()


def test_auto_wired_plugin_flow(tmp_path):
  """
  Scenario: Define a custom plugin 'MagicSwap' that auto-wires itself to
  a new operation 'MagicOp'.
  Input: `torch.magic(x)`
  Output: `jax.magic_swapped(x)`
  """

  # 1. Define the Plugin with Auto-Wire Metadata
  @register_hook(
    trigger="magic_swap",
    auto_wire={
      "ops": {
        "MagicOp": {
          "std_args": ["x"],
          "description": "Auto-wired magic operation",
          "variants": {
            "torch": {"api": "torch.magic"},
            "jax": {"api": "jax.magic_swapped", "requires_plugin": "magic_swap"},
          },
        }
      }
    },
  )
  def magic_plugin(node: cst.Call, ctx: HookContext) -> cst.Call:
    # Simple renaming logic
    # Valid CST construction for dotted name string 'jax.numpy.magic_swapped'
    # We use a helper function pattern usually, but manual here for verification

    attr_chain = cst.Attribute(
      value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")), attr=cst.Name("magic_swapped")
    )

    # Just swap the function name for simple verification
    # Must use simple Name if passing to func=...
    # For test, we expect 'wired_success(x)' string match
    return node.with_changes(func=cst.Name("wired_success"))

  # 2. Initialize Manager (Should hydrate from registry)
  # We patch resolve methods to point to empty dirs so no JSONs interfere
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=tmp_path):
      # Patch available_frameworks to prevent standard adapter loading noise
      with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
        mgr = SemanticsManager()

  # 3. Verify Manager State
  assert "MagicOp" in mgr.data
  assert mgr.data["MagicOp"]["variants"]["jax"]["requires_plugin"] == "magic_swap"
  # Verify Reverse Index matches torch.magic -> MagicOp
  lookup = mgr.get_definition("torch.magic")
  assert lookup is not None
  assert lookup[0] == "MagicOp"

  # 4. End-to-End AST Transformation
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)

  code = "res = torch.magic(data)"
  result = engine.run(code)

  assert result.success
  # The plugin changes func name to 'wired_success'
  assert "wired_success(data)" in result.code
