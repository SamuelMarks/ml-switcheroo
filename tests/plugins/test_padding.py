"""Test suite for the Padding module."""

from unittest.mock import MagicMock

import libcst as _cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter

cst = _cst


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing.

  Returns:
      PivotRewriter: A mock rewriter instance.
  """
  hooks.register_hook("padding_converter")(transform_padding)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  jax_variant: dict[str, JsonValue] = {"api": "jnp.pad", "requires_plugin": "padding_converter"}
  pad_def: dict[str, JsonValue] = {
    "variants": {
      "torch": {"api": "torch.nn.functional.pad"},
      "jax": jax_variant,
    }
  }
  mgr.get_definition.side_effect = lambda n: ("Pad", pad_def) if "pad" in n else None

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in padding tests.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        dict[str, JsonValue] | None: Variant definition.
    """
    if aid == "Pad" and fw == "jax":
      return jax_variant
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Pad": pad_def}
  mgr.is_verified.return_value = True

  def get_config(fw: str) -> dict[str, PluginTraits]:
    """Gets configuration.

    Args:
        fw (str): Target framework string.

    Returns:
        dict[str, PluginTraits]: The configuration dict.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_padding_2d_nchw(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jnp.pad` for padding 2d nchw.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "jnp.pad" in res
  assert "((0,0),(0,0),(3,4),(1,2))" in res.replace(" ", "")


def test_padding_passthrough_missing(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `F.pad` for padding passthrough missing.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.target_framework = "unknown"
  rewriter.context.hook_context.target_framework = "unknown"
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "F.pad" in res


def test_padding_missing_args(rewriter: PivotRewriter) -> None:
  """Verifies behavior when there are fewer than 2 arguments.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "y = F.pad(x)"
  res = rewrite_code(rewriter, code)
  assert "F.pad(x)" in res


def test_padding_missing_comma(rewriter: PivotRewriter) -> None:
  """Verifies the comma is added when it is missing on the input arg.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
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
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  ctx.lookup_api.return_value = "jnp.pad"
  res = transform_padding(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.args[0].comma, cst.Comma)
  assert cst.Module([]).code_for_node(res).replace(" ", "") == "jnp.pad(x,((0,0),(0,0),(3,4),(1,2)))"


def test_padding_keywords_prenested_and_6d() -> None:
  """Verify keyword pad/input args, pre-nested pair tuples, and 6-element NCDHW tuples."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  ctx.lookup_api.return_value = "jnp.pad"

  kw_6d = cst.ensure_type(
    cst.parse_expression("pad(pad=(1, 2, 3, 4, 5, 6), input=x, mode='constant', *extra)"),
    cst.Call,
  )
  res_6d = transform_padding(kw_6d, ctx)
  code_6d = cst.Module(body=[]).code_for_node(res_6d).replace(" ", "")
  assert code_6d == "jnp.pad(x,((0,0),(0,0),(5,6),(3,4),(1,2)))"

  nested = cst.ensure_type(cst.parse_expression("pad(x, ((0, 0), (1, 2)))"), cst.Call)
  res_nested = transform_padding(nested, ctx)
  assert cst.Module(body=[]).code_for_node(res_nested).replace(" ", "") == "jnp.pad(x,((0,0),(1,2)))"


def test_no_semantics(rewriter: PivotRewriter) -> None:
  """Verifies behavior when ctx.semantics is None.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  rewriter.ctx.semantics = None
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_config(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when get_framework_config returns None.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(rewriter.ctx.semantics, "get_framework_config", MagicMock(return_value=None), raising=False)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_traits(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when plugin_traits is missing.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(
    rewriter.ctx.semantics, "get_framework_config", MagicMock(return_value={"plugin_traits": None}), raising=False
  )
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_traits_is_dict(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when traits is a dict.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(
    rewriter.ctx.semantics,
    "get_framework_config",
    MagicMock(return_value={"plugin_traits": {"has_numpy_compatible_arrays": True}}),
    raising=False,
  )
  res = transform_padding(call_node, rewriter.ctx)
  assert "jnp.pad" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code


def test_traits_no_attr(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when traits object lacks has_numpy_compatible_arrays.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """

  class _DummyTraits:
    """Dummy traits without has_numpy_compatible_arrays."""

  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(
    rewriter.ctx.semantics,
    "get_framework_config",
    MagicMock(return_value={"plugin_traits": _DummyTraits()}),
    raising=False,
  )
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_no_target_api(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when lookup_api fails.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  code = "y = F.pad(x, (1, 2, 3, 4))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value=None), raising=False)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_not_tuple(rewriter: PivotRewriter) -> None:
  """Verifies behavior when padding arg is not a tuple.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "y = F.pad(x, pad_val)"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_pad_wrong_length(rewriter: PivotRewriter) -> None:
  """Verifies behavior when padding tuple is not length 4.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code = "y = F.pad(x, (1, 2))"
  stmt = cst.ensure_type(cst.parse_module(code).body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  call_node = cst.ensure_type(assign.value, cst.Call)
  res = transform_padding(call_node, rewriter.ctx)
  assert res is call_node


def test_padding_no_conf() -> None:
  """Verify `Call` handles padding no conf as expected."""
  node = cst.Call(func=cst.Name("pad"))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {}
  res = transform_padding(node, ctx)
  assert res is node


def test_padding_zero_args(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify padding plugin returns original node when called with fewer than 2 arguments.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  semantics = MagicMock()
  semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="jax.numpy.pad"), raising=False)
  node = cst.ensure_type(cst.parse_expression("pad(x)"), cst.Call)
  assert transform_padding(node, ctx) is node
