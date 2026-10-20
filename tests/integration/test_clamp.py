"""Test suite for the Clamp module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
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
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  clamp_def = {
    "std_args": ["input", "min", "max"],
    "variants": {
      "torch": {"api": "torch.clamp"},
      "jax": {"api": "jax.numpy.clip", "args": {"min": "a_min", "max": "a_max", "input": "a"}},
    },
  }
  mgr.get_definition.side_effect = lambda n: ("Clamp", clamp_def) if "clamp" in n or "clip" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: (
    clamp_def["variants"]["jax"] if aid == "Clamp" and fw == "jax" else None  # type: ignore[index]
  )
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Clamp": clamp_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clamp_keyword_rename(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of clamp keyword rename.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.clamp(x, min=0.0, max=1.0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0.0" in res
  assert "a_max=1.0" in res
  assert " min=" not in res


def test_clip_alias(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of clip alias.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.clip(x, 0, 1)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res


def test_method_clamp(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of method clamp.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = x.clamp(min=0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0" in res
