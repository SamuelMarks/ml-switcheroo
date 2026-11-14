"""Test suite for the Squeeze module."""

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
  mod = cst.parse_module(code)
  return str(rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for squeeze tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  squeeze_def: dict[str, JsonValue] = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.squeeze"}, "jax": {"api": "jax.numpy.squeeze", "args": {"dim": "axis"}}},
  }
  unsqueeze_def: dict[str, JsonValue] = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.unsqueeze"}, "jax": {"api": "jax.numpy.expand_dims", "args": {"dim": "axis"}}},
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "unsqueeze" in name:
      return ("Unsqueeze", unsqueeze_def)
    if "squeeze" in name:
      return ("Squeeze", squeeze_def)
    return None

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `unsqueeze_def.get` to support resolve in squeeze tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if aid == "Unsqueeze" and fw == "jax":
      variants_u = unsqueeze_def.get("variants")
      if isinstance(variants_u, dict):
        jax_u = variants_u.get("jax")
        if isinstance(jax_u, dict):
          return jax_u
      return {}
    if aid == "Squeeze" and fw == "jax":
      variants_s = squeeze_def.get("variants")
      if isinstance(variants_s, dict):
        jax_s = variants_s.get("jax")
        if isinstance(jax_s, dict):
          return jax_s
      return {}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Squeeze": squeeze_def, "Unsqueeze": unsqueeze_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_unsqueeze_mapping(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.expand_dims` for unsqueeze mapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.unsqueeze(x, dim=1)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "axis=1" in res
  assert "dim=" not in res


def test_squeeze_mapping(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.squeeze` for squeeze mapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.squeeze(x, dim=2)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.squeeze" in res
  assert "axis=2" in res


def test_method_to_function_unsqueeze(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.expand_dims` for method to function unsqueeze.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = x.unsqueeze(0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "(x, 0)" in res
