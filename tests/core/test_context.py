"""Test suite for the Context Gap module."""

import typing

from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext


def test_default_arg_injector() -> None:
  """Verifies the behavior of default argument injector."""
  ctx_default: typing.Any = RewriterContext(MagicMock(), None)
  assert ctx_default.config is not None
  ctx: typing.Any = RewriterContext(MagicMock(), MagicMock())
  sig_ctx: typing.Any = SignatureContext()
  ctx.signature_stack.append(sig_ctx)
  ctx._default_arg_injector("x", "int")
  assert ("x", "int") in sig_ctx.injected_args
  ctx._default_arg_injector("x", "int")
  assert len(sig_ctx.injected_args) == 1


def test_default_preamble_injector() -> None:
  """Verifies the behavior of default preamble injector."""
  ctx: typing.Any = RewriterContext(MagicMock(), MagicMock())
  ctx._default_preamble_injector("x = 1")
  assert "x = 1" in ctx.module_preamble
  assert "x = 1" in ctx._satisfied_preamble_injections
  ctx._default_preamble_injector("x = 1")
  assert len(ctx.module_preamble) == 1
  sig_ctx: typing.Any = SignatureContext()
  ctx.signature_stack.append(sig_ctx)
  ctx._default_preamble_injector("y = 2")
  assert "y = 2" in sig_ctx.preamble_stmts
  ctx._default_preamble_injector("y = 2")
  assert len(sig_ctx.preamble_stmts) == 1
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble


def test_hydrate_source_aliases_exception() -> None:
  """Verifies the behavior of hydrate source aliases correctly handling an exception."""
  semantics: typing.Any = MagicMock()
  semantics.get_framework_config.side_effect = KeyError("err")
  config: typing.Any = MagicMock()
  ctx: typing.Any = RewriterContext(semantics, config)
  _res: typing.Any = ctx._hydrate_source_aliases()
  assert _res is None
  assert ctx.alias_map == {}


def test_hydrate_source_aliases_pydantic() -> None:
  """Verifies the behavior of hydrate source aliases pydantic."""
  semantics: typing.Any = MagicMock()
  alias_info: typing.Any = MagicMock()
  alias_info.model_dump.return_value = {"name": "jax"}
  semantics.get_framework_config.return_value = {"alias": alias_info}
  config: typing.Any = MagicMock()
  ctx: typing.Any = RewriterContext(semantics, config)
  assert ctx.alias_map["jax"] == "jax"


def test_hydrate_source_aliases_no_name() -> None:
  """Verifies the behavior of hydrate source aliases no name."""
  semantics: typing.Any = MagicMock()
  semantics.get_framework_config.return_value = {"alias": {}}
  config: typing.Any = MagicMock()
  ctx: typing.Any = RewriterContext(semantics, config)
  assert "jax" not in ctx.alias_map


def test_hydrate_source_aliases_none() -> None:
  """Verifies the behavior of hydrate source aliases none."""
  semantics: typing.Any = MagicMock()
  semantics.get_framework_config.return_value = None
  config: typing.Any = MagicMock()
  ctx: typing.Any = RewriterContext(semantics, config)
  assert "jax" not in ctx.alias_map
