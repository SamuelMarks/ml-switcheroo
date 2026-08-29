"""Test suite for the Batch Norm module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["batch_norm_unwrap"] = transform_batch_norm
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  bn_def: dict[str, typing.Any] = {
    "variants": {
      "torch": {"api": "torch.nn.BatchNorm2d"},
      "jax": {"api": "flax.nnx.BatchNorm", "requires_plugin": "batch_norm_unwrap"},
    }
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    return ("BatchNorm", bn_def) if "BatchNorm" in name or "bn" in name else None

  def get_fw_config(fw: str) -> dict[str, typing.Any]:
    """Gets framework configuration."""
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_state=True)}
    return {}

  mgr.get_definition.side_effect = get_def
  mgr.get_framework_config.side_effect = get_fw_config

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    return typing.cast(dict[str, typing.Any], bn_def["variants"]["jax"]) if fw == "jax" and aid == "BatchNorm" else None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"BatchNorm": bn_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_bn_injection_and_unwrap(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn injection and unwrap."""
  code: str = "y = self.bn(x)"
  res: str = rewrite_code(rewriter, code)
  clean: str = res.replace(" ", "")
  assert "use_running_average=nottraining" in clean
  assert "mutable=['batch_stats']" in clean
  assert res.strip().endswith(")[0]")


def test_bn_nested_expression(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn nested expression."""
  code: str = "y = F.relu(self.bn(x))"
  res: str = rewrite_code(rewriter, code)
  assert "self.bn(x" in res
  assert "[0])" in res


def test_bn_preserve_existing_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn preserve existing arguments."""
  code: str = "y = self.bn(x, other=1)"
  res: str = rewrite_code(rewriter, code)
  assert "other=1" in res
  assert "use_running_average" in res
