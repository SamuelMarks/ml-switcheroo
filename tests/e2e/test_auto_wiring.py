"""Test suite for the Auto Wiring module."""

from pathlib import Path
from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.semantics.manager import SemanticsManager, clear_semantics_cache


@pytest.fixture(autouse=True)
def clean_env() -> None:
  """Create a configured `None` fixture for auto wiring tests."""


def test_auto_wired_plugin_flow(tmp_path: Path) -> None:
  """Verify `clear_semantics_cache` emits `MagicOp` for auto wired plugin flow.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  clear_semantics_cache()

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
    """Execute `register_hook` to support magic plugin in auto wiring tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.Call` from magic plugin.
    """
    cst.Attribute(value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")), attr=cst.Name("magic_swapped"))
    return node.with_changes(func=cst.Name("wired_success"))

  with (
    patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=tmp_path),
    patch("ml_switcheroo.semantics.paths.resolve_snapshots_dir", return_value=tmp_path),
    patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]),
  ):
    mgr = SemanticsManager()
  assert "MagicOp" in mgr.data
  variants = mgr.data["MagicOp"].get("variants")
  assert isinstance(variants, dict)
  jax_variant = variants.get("jax")
  assert isinstance(jax_variant, dict)
  assert jax_variant.get("requires_plugin") == "magic_swap"
  lookup = mgr.get_definition("torch.magic")
  assert lookup is not None
  assert lookup[0] == "MagicOp"
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)
  code: str = "res = torch.magic(data)"
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "wired_success(data)" in result.code
