"""Test suite for the Batch Norm module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
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
  hooks._HOOKS["batch_norm_unwrap"] = transform_batch_norm  # type: ignore[assignment]
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  bn_def = {
    "variants": {
      "torch": {"api": "torch.nn.BatchNorm2d"},
      "jax": {"api": "flax.nnx.BatchNorm", "requires_plugin": "batch_norm_unwrap"},
    }
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return ("BatchNorm", bn_def) if "BatchNorm" in name or "bn" in name else None  # type: ignore[return-value]

  def get_fw_config(fw: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_state=True)}  # type: ignore[dict-item]
    return {}

  mgr.get_definition.side_effect = get_def
  mgr.get_framework_config.side_effect = get_fw_config

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return (
      (bn_def["variants"]["jax"] if isinstance(bn_def["variants"]["jax"], dict) else {})  # type: ignore[return-value, redundant-expr]
      if fw == "jax" and aid == "BatchNorm"
      else None
    )

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"BatchNorm": bn_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_bn_injection_and_unwrap(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn injection and unwrap.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = self.bn(x)"
  res: str = rewrite_code(rewriter, code)
  clean: str = res.replace(" ", "")
  assert "use_running_average=nottraining" in clean
  assert "mutable=['batch_stats']" in clean
  assert res.strip().endswith(")[0]")


def test_bn_nested_expression(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn nested expression.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = F.relu(self.bn(x))"
  res: str = rewrite_code(rewriter, code)
  assert "self.bn(x" in res
  assert "[0])" in res


def test_bn_preserve_existing_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn preserve existing arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = self.bn(x, other=1)"
  res: str = rewrite_code(rewriter, code)
  assert "other=1" in res
  assert "use_running_average" in res


def test_bn_no_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn when called with no arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = self.bn()"
  res: str = rewrite_code(rewriter, code)
  assert "use_running_average" in res
  assert "mutable" in res


def test_bn_already_has_comma(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when argument already has a comma.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = self.bn(use_running_average=True,)"
  res: str = rewrite_code(rewriter, code)
  assert "mutable" in res
