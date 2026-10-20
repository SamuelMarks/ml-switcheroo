"""Test suite for the Clipping module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.clipping import transform_grad_clipping
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  hooks._HOOKS["grad_clipper"] = transform_grad_clipping  # type: ignore[assignment]
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  clip_def = {
    "variants": {
      "torch": {"api": "torch.nn.utils.clip_grad_norm_"},
      "jax": {"api": "optax.clip_by_global_norm", "requires_plugin": "grad_clipper"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("ClipGrads", clip_def) if "clip_grad" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: clip_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"ClipGrads": clip_def}
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_functional_state=True)}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clip_transform(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of clip transform.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "torch.nn.utils.clip_grad_norm_(grads, 1.0)"
  res: str = rewrite_code(rewriter, code)
  assert "optax.clip_by_global_norm(1.0)" in res
  assert ".update(grads, None)" in res
  assert ")[0]" in res.strip()


def test_clip_with_variable_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of clip with variable arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "clip_grad_norm_(g, max_val)"
  res: str = rewrite_code(rewriter, code)
  assert "clip_by_global_norm(max_val)" in res
  assert "update(g," in res


def test_ignores_if_traits_missing(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignores if traits missing.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_functional_state=False)}  # type: ignore[attr-defined]
  code: str = "torch.nn.utils.clip_grad_norm_(g, 1.0)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" not in res


def test_clip_with_missing_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of clip with missing arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "torch.nn.utils.clip_grad_norm_(grads)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" not in res
