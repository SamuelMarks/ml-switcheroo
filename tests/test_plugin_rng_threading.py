"""Test suite for plugin rng threading."""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.plugins.rng_threading import _remove_generator_arg, inject_prng_threading


def test_remove_generator_arg() -> None:
  """Verify remove generator arg behavior."""
  # Has generator
  args: typing.List[cst.Arg] = [
    cst.Arg(value=cst.Name("x")),
    cst.Arg(keyword=cst.Name("generator"), value=cst.Name("gen")),
  ]
  cleaned: typing.List[cst.Arg] = _remove_generator_arg(args)
  assert len(cleaned) == 1
  assert getattr(cleaned[0].value, "value", None) == "x"

  # No generator
  args2: typing.List[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(keyword=cst.Name("other"), value=cst.Name("gen"))]
  cleaned2: typing.List[cst.Arg] = _remove_generator_arg(args2)
  assert len(cleaned2) == 2


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_enabled(mock_get_adapter: MagicMock) -> None:
  """Verify inject prng threading enabled behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.get_rng_split_syntax.return_value = "rng, key = jax.random.split(rng)"
  mock_get_adapter.return_value = mock_adapter

  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = True
  ctx.raw_config.side_effect = lambda k, default: default
  ctx.target_fw = "jax"

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x, generator=gen)"), cst.Call)

  new_node: cst.BaseExpression = inject_prng_threading(node, ctx)

  ctx.inject_signature_arg.assert_called_once_with("rng")
  ctx.inject_preamble.assert_called_once_with("rng, key = jax.random.split(rng)")

  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code  # new_node is a Call here
  assert code.strip() == "dropout(x, key=key)"


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_disabled(mock_get_adapter: MagicMock) -> None:
  """Verify inject prng threading disabled behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = False

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x)"), cst.Call)

  new_node: cst.BaseExpression = inject_prng_threading(node, ctx)

  assert new_node is node
  mock_get_adapter.assert_not_called()


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_no_adapter_or_split(mock_get_adapter: MagicMock) -> None:
  """Verify inject prng threading no adapter or split behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  # Test adapter is None
  mock_get_adapter.return_value = None

  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = True
  ctx.raw_config.side_effect = lambda k, default: default

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x)"), cst.Call)
  inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called

  # Test split is None or "pass"
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.get_rng_split_syntax.return_value = "pass"
  mock_get_adapter.return_value = mock_adapter

  inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called

  mock_adapter.get_rng_split_syntax.return_value = None
  inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called
