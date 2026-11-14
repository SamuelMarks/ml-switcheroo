"""Test suite for plugin rng threading."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.plugins.rng_threading import _remove_generator_arg, inject_prng_threading


def test_remove_generator_arg() -> None:
  """Verify `_remove_generator_arg` returns `1` across plugin RNG threading inputs."""
  # Has generator
  args: list[cst.Arg] = [
    cst.Arg(value=cst.Name("x")),
    cst.Arg(keyword=cst.Name("generator"), value=cst.Name("gen")),
  ]
  cleaned: list[cst.Arg] = _remove_generator_arg(args)
  assert len(cleaned) == 1
  assert cst.ensure_type(cleaned[0].value, cst.Name).value == "x"

  # No generator
  args2: list[cst.Arg] = [cst.Arg(value=cst.Name("x")), cst.Arg(keyword=cst.Name("other"), value=cst.Name("gen"))]
  cleaned2: list[cst.Arg] = _remove_generator_arg(args2)
  assert len(cleaned2) == 2


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_enabled(mock_get_adapter: MagicMock) -> None:
  """Verify `ensure_type` handles inject prng threading enabled as expected.

  Args:
      mock_get_adapter: Mock object for get adapter.
  """
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.get_rng_split_syntax.return_value = "rng, key = jax.random.split(rng)"
  mock_get_adapter.return_value = mock_adapter

  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = True
  ctx.raw_config.side_effect = lambda k, default: default
  ctx.target_framework = "jax"

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x, generator=gen)"), cst.Call)

  new_node: cst.BaseExpression = inject_prng_threading(node, ctx)

  ctx.inject_signature_arg.assert_called_once_with("rng")
  ctx.inject_preamble.assert_called_once_with("rng, key = jax.random.split(rng)")

  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code  # new_node is a Call here
  assert code.strip() == "dropout(x, key=key)"


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_disabled(mock_get_adapter: MagicMock) -> None:
  """Verify `ensure_type` handles inject prng threading disabled as expected.

  Args:
      mock_get_adapter: Mock object for get adapter.
  """
  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = False

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x)"), cst.Call)

  new_node: cst.BaseExpression = inject_prng_threading(node, ctx)

  assert new_node is node
  mock_get_adapter.assert_not_called()


@patch("ml_switcheroo.plugins.rng_threading.get_adapter")
def test_inject_prng_threading_no_adapter_or_split(mock_get_adapter: MagicMock) -> None:
  """Verify `ensure_type` handles inject prng threading no adapter or split as expected.

  Args:
      mock_get_adapter: Mock object for get adapter.
  """
  # Test adapter is None
  mock_get_adapter.return_value = None

  ctx: MagicMock = MagicMock()
  ctx.plugin_traits.requires_explicit_rng = True
  ctx.raw_config.side_effect = lambda k, default: default

  node: cst.Call = cst.ensure_type(cst.parse_expression("dropout(x)"), cst.Call)
  res_no_adapter = inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called
  assert cst.Module([]).code_for_node(res_no_adapter) == "dropout(x, key=key)"

  # Test split is None or "pass"
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.get_rng_split_syntax.return_value = "pass"
  mock_get_adapter.return_value = mock_adapter

  res_pass_split = inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called
  assert cst.Module([]).code_for_node(res_pass_split) == "dropout(x, key=key)"

  mock_adapter.get_rng_split_syntax.return_value = None
  res_none_split = inject_prng_threading(node, ctx)
  assert not ctx.inject_preamble.called
  assert cst.Module([]).code_for_node(res_none_split) == "dropout(x, key=key)"
  assert ctx.inject_signature_arg.call_count == 3
  assert ctx.inject_preamble.call_count == 0
