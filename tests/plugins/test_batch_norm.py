"""Test suite for the Batch Norm module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
from ml_switcheroo.semantics.schema import PluginTraits
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
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for batch norm tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("batch_norm_unwrap")(transform_batch_norm)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  jax_variant: dict[str, JsonValue] = {"api": "flax.nnx.BatchNorm", "requires_plugin": "batch_norm_unwrap"}
  bn_def: dict[str, JsonValue] = {
    "variants": {
      "torch": {"api": "torch.nn.BatchNorm2d"},
      "jax": jax_variant,
    }
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    return ("BatchNorm", bn_def) if "BatchNorm" in name or "bn" in name else None

  def get_fw_config(fw: str) -> dict[str, PluginTraits]:
    """Gets framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, PluginTraits]` from get fw config.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_state=True)}
    return {}

  mgr.get_definition.side_effect = get_def
  mgr.get_framework_config.side_effect = get_fw_config

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in batch norm tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if fw == "jax" and aid == "BatchNorm":
      return jax_variant
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"BatchNorm": bn_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_bn_injection_and_unwrap(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `mutable=['batch_stats']` for bn injection and unwrap.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = self.bn(x)"
  res: str = rewrite_code(rewriter, code)
  clean: str = res.replace(" ", "")
  assert "use_running_average=nottraining" in clean
  assert "mutable=['batch_stats']" in clean
  assert res.strip().endswith(")[0]")


def test_bn_nested_expression(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `self.bn(x` for bn nested expression.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = F.relu(self.bn(x))"
  res: str = rewrite_code(rewriter, code)
  assert "self.bn(x" in res
  assert "[0])" in res


def test_bn_preserve_existing_args(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `other=1` for bn preserve existing args.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = self.bn(x, other=1)"
  res: str = rewrite_code(rewriter, code)
  assert "other=1" in res
  assert "use_running_average" in res


def test_bn_no_args(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `use_running_average` for bn no args.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = self.bn()"
  res: str = rewrite_code(rewriter, code)
  assert "use_running_average" in res
  assert "mutable" in res


def test_bn_already_has_comma(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when argument already has a comma.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = self.bn(use_running_average=True,)"
  res: str = rewrite_code(rewriter, code)
  assert "mutable" in res
