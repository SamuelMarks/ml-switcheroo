from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext


def test_default_arg_injector():
  ctx = RewriterContext(MagicMock(), MagicMock())
  sig_ctx = SignatureContext()
  ctx.signature_stack.append(sig_ctx)
  ctx._default_arg_injector("x", "int")
  assert ("x", "int") in sig_ctx.injected_args
  # duplicate
  ctx._default_arg_injector("x", "int")
  assert len(sig_ctx.injected_args) == 1


def test_default_preamble_injector():
  ctx = RewriterContext(MagicMock(), MagicMock())

  # Empty stack
  ctx._default_preamble_injector("x = 1")
  assert "x = 1" in ctx.module_preamble
  assert "x = 1" in ctx._satisfied_preamble_injections

  # Duplicate empty stack
  ctx._default_preamble_injector("x = 1")
  assert len(ctx.module_preamble) == 1

  # With signature stack
  sig_ctx = SignatureContext()
  ctx.signature_stack.append(sig_ctx)

  ctx._default_preamble_injector("y = 2")
  assert "y = 2" in sig_ctx.preamble_stmts

  # duplicate
  ctx._default_preamble_injector("y = 2")
  assert len(sig_ctx.preamble_stmts) == 1

  # Import
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble


def test_hydrate_source_aliases_exception():
  semantics = MagicMock()
  semantics.get_framework_config.side_effect = Exception("err")
  config = MagicMock()
  ctx = RewriterContext(semantics, config)
  # Shouldn't raise
  ctx._hydrate_source_aliases()


def test_hydrate_source_aliases_pydantic():
  semantics = MagicMock()
  alias_info = MagicMock()
  alias_info.model_dump.return_value = {"name": "jax"}
  semantics.get_framework_config.return_value = {"alias": alias_info}
  config = MagicMock()
  ctx = RewriterContext(semantics, config)
  assert ctx.alias_map["jax"] == "jax"


def test_hydrate_source_aliases_no_name():
  semantics = MagicMock()
  semantics.get_framework_config.return_value = {"alias": {}}
  config = MagicMock()
  ctx = RewriterContext(semantics, config)
  assert "jax" not in ctx.alias_map


def test_hydrate_source_aliases_none():
  semantics = MagicMock()
  semantics.get_framework_config.return_value = None
  config = MagicMock()
  ctx = RewriterContext(semantics, config)
  assert "jax" not in ctx.alias_map
