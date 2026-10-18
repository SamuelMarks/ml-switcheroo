"""Test suite for the Padding module."""

import typing

from typing import Any
from unittest.mock import MagicMock

import libcst as _cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter

cst: typing.Any = _cst


def rewrite_code(rewriter: typing.Any, code: typing.Any) -> typing.Any:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> typing.Any:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["padding_converter"] = transform_padding
  hooks._PLUGINS_LOADED = True
  mgr: typing.Any = MagicMock()
  pad_def: typing.Any = {
    "variants": {
      "torch": {"api": "torch.nn.functional.pad"},
      "jax": {"api": "jnp.pad", "requires_plugin": "padding_converter"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("Pad", pad_def) if "pad" in n else None

  def resolve(aid: typing.Any, fw: typing.Any) -> Any:
    """Resolves .

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        Any: Variant definition.
    """
    if aid == "Pad" and fw == "jax":
      return pad_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Pad": pad_def}
  mgr.is_verified.return_value = True

  def get_config(fw: typing.Any) -> typing.Any:
    """Gets configuration.

    Args:
        fw (str): Target framework string.

    Returns:
        Dict[str, Any]: The configuration dict.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  yield PivotRewriter(mgr, cfg)


def test_padding_2d_nchw(rewriter: typing.Any) -> None:
  """Verifies the behavior of padding 2d nchw.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "jnp.pad" in res
  assert "((0,0),(0,0),(3,4),(1,2))" in res.replace(" ", "")


def test_padding_passthrough_missing(rewriter: typing.Any) -> None:
  """Verifies the behavior of padding passthrough missing.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.target_framework = "unknown"
  rewriter.context.hook_context.target_fw = "unknown"
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "F.pad" in res


def test_padding_missing_args(rewriter: typing.Any) -> None:
  """Verifies behavior when there are fewer than 2 arguments.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x)"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "F.pad(x)" in res


def test_padding_missing_comma(rewriter: typing.Any) -> None:
  """Verifies the comma is added when it is missing on the input arg.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  # We construct a node directly to bypass cst parsing which normally adds commas
  node: typing.Any = cst.Call(
    func=cst.Name("pad"),
    args=[
      cst.Arg(cst.Name("x"), comma=cst.MaybeSentinel.DEFAULT),
      cst.Arg(
        cst.Tuple(
          [
            cst.Element(cst.Integer("1")),
            cst.Element(cst.Integer("2")),
            cst.Element(cst.Integer("3")),
            cst.Element(cst.Integer("4")),
          ]
        )
      ),
    ],
  )
  ctx: typing.Any = MagicMock()
  ctx.target_fw = "jax"
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  ctx.lookup_api.return_value = "jnp.pad"
  res: typing.Any = transform_padding(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.args[0].comma, cst.Comma)


def test_no_semantics(rewriter: typing.Any) -> None:
  """Verifies behavior when ctx.semantics is None.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics = None
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_config(rewriter: typing.Any) -> None:
  """Verifies behavior when get_framework_config returns None.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value=None)
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_traits(rewriter: typing.Any) -> None:
  """Verifies behavior when plugin_traits is missing.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value={"plugin_traits": None})
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_traits_is_dict(rewriter: typing.Any) -> None:
  """Verifies behavior when traits is a dict.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(
    return_value={"plugin_traits": {"has_numpy_compatible_arrays": True}}
  )
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert "jnp.pad" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code


def test_traits_no_attr(rewriter: typing.Any) -> None:
  """Verifies behavior when traits object lacks has_numpy_compatible_arrays.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value={"plugin_traits": object()})
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_target_api(rewriter: typing.Any) -> None:
  """Verifies behavior when lookup_api fails.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2, 3, 4))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_not_tuple(rewriter: typing.Any) -> None:
  """Verifies behavior when padding arg is not a tuple.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, pad_val)"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_wrong_length(rewriter: typing.Any) -> None:
  """Verifies behavior when padding tuple is not length 4.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: typing.Any = "y = F.pad(x, (1, 2))"
  call_node: typing.Any = cst.parse_module(code).body[0].body[0].value
  res: typing.Any = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


# --- Merged from test_padding_extra.py ---


def test_padding_no_conf() -> None:
  """Verifies the behavior of padding no conf."""
  node: typing.Any = cst.Call(func=cst.Name("pad"))
  semantics: typing.Any = MagicMock()
  ctx: typing.Any = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {}
  res: typing.Any = transform_padding(node, ctx)
  assert res is node
