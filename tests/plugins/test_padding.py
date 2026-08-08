"""Test suite for the Padding module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["padding_converter"] = transform_padding
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  pad_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.pad"},
      "jax": {"api": "jnp.pad", "requires_plugin": "padding_converter"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("Pad", pad_def) if "pad" in n else None

  def resolve(aid, fw):
    """Resolves ."""
    if aid == "Pad" and fw == "jax":
      return pad_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Pad": pad_def}
  mgr.is_verified.return_value = True

  def get_config(fw):
    """Gets configuration."""
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_padding_2d_nchw(rewriter):
  """Verifies the behavior of padding 2d nchw."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "jnp.pad" in res
  assert "((0,0),(0,0),(3,4),(1,2))" in res.replace(" ", "")


def test_padding_passthrough_missing(rewriter):
  """Verifies the behavior of padding passthrough missing."""
  rewriter.context.config.target_framework = "unknown"
  rewriter.context.hook_context.target_fw = "unknown"
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "F.pad" in res


def test_padding_missing_args(rewriter):
  """Verifies behavior when there are fewer than 2 arguments."""
  code = "y = F.pad(x)"
  res = rewrite_code(rewriter, code)
  assert "F.pad(x)" in res


def test_padding_missing_comma(rewriter):
  """Verifies the comma is added when it is missing on the input arg."""
  # We construct a node directly to bypass cst parsing which normally adds commas
  node = cst.Call(
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
  ctx = MagicMock()
  ctx.target_fw = "jax"
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  ctx.lookup_api.return_value = "jnp.pad"
  res = transform_padding(node, ctx)
  assert isinstance(res.args[0].comma, cst.Comma)


def test_no_semantics(rewriter):
  """Verifies behavior when ctx.semantics is None."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics = None
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_config(rewriter):
  """Verifies behavior when get_framework_config returns None."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value=None)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_traits(rewriter):
  """Verifies behavior when plugin_traits is missing."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value={"plugin_traits": None})
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_traits_is_dict(rewriter):
  """Verifies behavior when traits is a dict."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(
    return_value={"plugin_traits": {"has_numpy_compatible_arrays": True}}
  )
  res = transform_padding(call_node, rewriter.ctx)
  assert "jnp.pad" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code


def test_traits_no_attr(rewriter):
  """Verifies behavior when traits object lacks has_numpy_compatible_arrays."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.semantics.get_framework_config = MagicMock(return_value={"plugin_traits": object()})
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_target_api(rewriter):
  """Verifies behavior when lookup_api fails."""
  code = "y = F.pad(x, (1, 2, 3, 4))"
  call_node = cst.parse_module(code).body[0].body[0].value
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_not_tuple(rewriter):
  """Verifies behavior when padding arg is not a tuple."""
  code = "y = F.pad(x, pad_val)"
  call_node = cst.parse_module(code).body[0].body[0].value
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_wrong_length(rewriter):
  """Verifies behavior when padding tuple is not length 4."""
  code = "y = F.pad(x, (1, 2))"
  call_node = cst.parse_module(code).body[0].body[0].value
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node
