"""Test suite for the Auto Wiring module."""

import pytest
import libcst as cst
from unittest.mock import patch
from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture(autouse=True)
def clean_env():
  """Helper to clean environment."""
  pass
  yield
  pass


def test_auto_wired_plugin_flow(tmp_path):
  """Verifies the behavior of auto wired plugin flow."""

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
    """Helper to magic plugin."""
    cst.Attribute(value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")), attr=cst.Name("magic_swapped"))
    return node.with_changes(func=cst.Name("wired_success"))

  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=tmp_path):
    with patch("ml_switcheroo.semantics.paths.resolve_snapshots_dir", return_value=tmp_path):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        mgr = SemanticsManager()
  assert "MagicOp" in mgr.data
  assert mgr.data["MagicOp"]["variants"]["jax"]["requires_plugin"] == "magic_swap"
  lookup = mgr.get_definition("torch.magic")
  assert lookup is not None
  assert lookup[0] == "MagicOp"
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)
  code = "res = torch.magic(data)"
  result = engine.run(code)
  assert result.success
  assert "wired_success(data)" in result.code
