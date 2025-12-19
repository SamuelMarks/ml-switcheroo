"""
Tests for Gradient Clipping Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.clipping import transform_grad_clipping


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["grad_clipper"] = transform_grad_clipping
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define Clipping
  clip_def = {
    "variants": {
      "torch": {"api": "torch.nn.utils.clip_grad_norm_"},
      "jax": {
        "api": "optax.clip_by_global_norm",  # Nominal
        "requires_plugin": "grad_clipper",
      },
    }
  }

  mgr.get_definition.side_effect = lambda n: ("ClipGrads", clip_def) if "clip_grad" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: clip_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"ClipGrads": clip_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clip_transform(rewriter):
  """
  Input: torch.nn.utils.clip_grad_norm_(grads, 1.0)
  Output: optax.clip_by_global_norm(1.0).update(grads, None)[0]
  """
  code = "torch.nn.utils.clip_grad_norm_(grads, 1.0)"
  res = rewrite_code(rewriter, code)

  assert "optax.clip_by_global_norm(1.0)" in res
  assert ".update(grads, None)" in res
  assert ")[0]" in res.strip()


def test_clip_with_variable_args(rewriter):
  """
  Input: clip_grad_norm_(g, max_norm)
  Output: optax... (max_norm) ... (g, None)[0]
  """
  code = "clip_grad_norm_(g, max_val)"
  res = rewrite_code(rewriter, code)

  assert "clip_by_global_norm(max_val)" in res
  assert "update(g," in res


def test_ignores_if_target_torch(rewriter):
  rewriter.ctx._runtime_config.target_framework = "torch"
  rewriter.ctx.target_fw = "torch"

  code = "torch.nn.utils.clip_grad_norm_(g, 1.0)"
  res = rewrite_code(rewriter, code)

  # Should perform basic mapping if defined in definitions, but plugin won't run logic
  # Since rewrite_code invokes plugin directly logic in tests usually? No, via rewriter.
  # PivotRewriter would rename API if strict map, but our mock might be skipped if we assume
  # Torch->Torch.
  # Plugin logic explicitly returns node if target != jax/flax.

  assert "optax" not in res
