"""Test suite for the Ckpt Keys module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.checkpoint_keys import KEY_MAPPER_SOURCE, transform_checkpoint_keys
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  return str(rewriter.convert(tree).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("checkpoint_mapper")(transform_checkpoint_keys)
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  load_def = {
    "variants": {
      "torch": {"api": "torch.nn.Module.load_state_dict"},
      "jax": {"api": "CustomKeyMapper", "requires_plugin": "checkpoint_mapper"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("LoadState", load_def) if "load_state_dict" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: load_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"LoadState": load_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_load_state_dict_rewrite(rewriter: PivotRewriter) -> None:
  """Loads state dictionary rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "model.load_state_dict(sd)"
  res: str = rewrite_code(rewriter, code)
  assert "KeyMapper.from_torch" in res
  assert "(sd)" in res
  assert "strict" not in res


def test_load_state_dict_kwargs(rewriter: PivotRewriter) -> None:
  """Loads state dictionary keyword arguments.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x.load_state_dict(state_dict=y, strict=False)"
  res: str = rewrite_code(rewriter, code)
  assert "KeyMapper.from_torch(y)" in res.replace(" ", "")


def test_mapper_source_availability() -> None:
  """Verifies the behavior of mapper source availability."""
  assert "class KeyMapper" in KEY_MAPPER_SOURCE
  assert "map_name" in KEY_MAPPER_SOURCE
  assert "map_value" in KEY_MAPPER_SOURCE
  assert "transpose" in KEY_MAPPER_SOURCE
  assert "replace" in KEY_MAPPER_SOURCE
