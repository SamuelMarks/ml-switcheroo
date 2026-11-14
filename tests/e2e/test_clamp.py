"""Test suite for the Clamp module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter(monkeypatch: pytest.MonkeyPatch) -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for clamp tests.

  Args:
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  jax_variant: dict[str, JsonValue] = {
    "api": "jax.numpy.clip",
    "args": {"min": "a_min", "max": "a_max", "input": "a"},
  }
  clamp_def: dict[str, JsonValue] = {
    "std_args": ["input", "min", "max"],
    "variants": {
      "torch": {"api": "torch.clamp"},
      "jax": jax_variant,
    },
  }
  mgr.get_definition.side_effect = lambda n: ("Clamp", clamp_def) if "clamp" in n or "clip" in n else None
  monkeypatch.setattr(
    mgr.resolve_variant,
    "side_effect",
    (lambda aid, fw: jax_variant if aid == "Clamp" and fw == "jax" else None),
    raising=False,
  )
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Clamp": clamp_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clamp_keyword_rename(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.clip` for clamp keyword rename.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.clamp(x, min=0.0, max=1.0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0.0" in res
  assert "a_max=1.0" in res
  assert " min=" not in res


def test_clip_alias(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.clip` for clip alias.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.clip(x, 0, 1)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res


def test_method_clamp(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.clip` for method clamp.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = x.clamp(min=0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0" in res
