"""Test suite for the Squeeze module."""

import typing
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
  mod = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(mod).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  squeeze_def: dict[str, typing.Any] = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.squeeze"}, "jax": {"api": "jax.numpy.squeeze", "args": {"dim": "axis"}}},
  }
  unsqueeze_def: dict[str, typing.Any] = {
    "std_args": ["input", "dim"],
    "variants": {"torch": {"api": "torch.unsqueeze"}, "jax": {"api": "jax.numpy.expand_dims", "args": {"dim": "axis"}}},
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "unsqueeze" in name:
      return ("Unsqueeze", unsqueeze_def)
    if "squeeze" in name:
      return ("Squeeze", squeeze_def)
    return None

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Unsqueeze" and fw == "jax":
      return typing.cast(dict[str, typing.Any], unsqueeze_def["variants"]["jax"])
    if aid == "Squeeze" and fw == "jax":
      return typing.cast(dict[str, typing.Any], squeeze_def["variants"]["jax"])
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Squeeze": squeeze_def, "Unsqueeze": unsqueeze_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_unsqueeze_mapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of unsqueeze mapping.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.unsqueeze(x, dim=1)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "axis=1" in res
  assert "dim=" not in res


def test_squeeze_mapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of squeeze mapping.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.squeeze(x, dim=2)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.squeeze" in res
  assert "axis=2" in res


def test_method_to_function_unsqueeze(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of method to function unsqueeze.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = x.unsqueeze(0)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.numpy.expand_dims" in res
  assert "(x, 0)" in res
