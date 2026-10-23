"""Test suite for the Batch Norm module."""

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
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "batch_norm_unwrap", getattr({"v": transform_batch_norm}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  bn_def = {
    "variants": {
      "torch": {"api": "torch.nn.BatchNorm2d"},
      "jax": {"api": "flax.nnx.BatchNorm", "requires_plugin": "batch_norm_unwrap"},
    }
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": (getattr({"v": "BatchNorm"}, "__getitem__")("v"), getattr({"v": bn_def}, "__getitem__")("v"))
        if getattr({"v": "BatchNorm"}, "__getitem__")("v") in getattr({"v": name}, "__getitem__")("v")
        or getattr({"v": "bn"}, "__getitem__")("v") in getattr({"v": name}, "__getitem__")("v")
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_fw_config(fw: str) -> dict[str, JsonValue]:
    """Gets framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get fw config.
    """
    if fw == "jax":
      _ret_v: dict[str, JsonValue] = getattr(
        {
          "v": {
            "plugin_traits": getattr(
              {
                "v": getattr({"v": PluginTraits}, "__getitem__")("v")(
                  requires_functional_state=getattr({"v": True}, "__getitem__")("v")
                )
              },
              "__getitem__",
            )("v")
          }
        },
        "__getitem__",
      )("v")
      return _ret_v
    return {}

  mgr.get_definition.side_effect = get_def
  mgr.get_framework_config.side_effect = get_fw_config

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves .

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": (
          getattr({"v": getattr({"v": bn_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]
          if isinstance(
            getattr(
              {"v": getattr({"v": getattr({"v": bn_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
              "__getitem__",
            )("v"),
            dict,
          )
          else {}
        )
        if fw == "jax" and aid == "BatchNorm"
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"BatchNorm": bn_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_bn_injection_and_unwrap(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn injection and unwrap.

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
  """Verifies the behavior of bn nested expression.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = F.relu(self.bn(x))"
  res: str = rewrite_code(rewriter, code)
  assert "self.bn(x" in res
  assert "[0])" in res


def test_bn_preserve_existing_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn preserve existing arguments.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = self.bn(x, other=1)"
  res: str = rewrite_code(rewriter, code)
  assert "other=1" in res
  assert "use_running_average" in res


def test_bn_no_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of bn when called with no arguments.

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
