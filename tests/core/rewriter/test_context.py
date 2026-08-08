"""Test suite for the Rewriter Context module."""

from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.context import RewriterContext, SignatureContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


def test_rewriter_context_inject_argument():
  """Verifies inject_argument correctly uses the signature stack."""
  semantics = MagicMock(spec=SemanticsManager)
  config = RuntimeConfig(target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)

  # No stack
  ctx._default_arg_injector("device", "str")

  # With stack
  sig_ctx = SignatureContext()
  ctx.signature_stack.append(sig_ctx)

  ctx._default_arg_injector("device", "str")
  assert len(sig_ctx.injected_args) == 1
  assert sig_ctx.injected_args[0] == ("device", "str")

  # Duplicate argument
  ctx._default_arg_injector("device", "str")
  assert len(sig_ctx.injected_args) == 1


def test_rewriter_context_preamble_injector():
  """Verifies _default_preamble_injector."""
  semantics = MagicMock(spec=SemanticsManager)
  config = RuntimeConfig(target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)

  # No stack -> module preamble
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble

  # Duplicate -> no op
  ctx._default_preamble_injector("import os")
  assert len(ctx.module_preamble) == 1

  # With stack
  sig_ctx = SignatureContext()
  ctx.signature_stack.append(sig_ctx)

  # Not an import -> preamble_stmts
  ctx._default_preamble_injector("x = 1")
  assert len(sig_ctx.preamble_stmts) == 1
  assert sig_ctx.preamble_stmts[0] == "x = 1"

  # Duplicate inside function -> no op
  ctx._default_preamble_injector("x = 1")
  assert len(sig_ctx.preamble_stmts) == 1

  # An import with stack -> module preamble
  ctx._default_preamble_injector("import sys")
  assert "import sys" in ctx.module_preamble


def test_rewriter_context_hydrate_aliases():
  """Verifies _hydrate_aliases exception and model_dump branches."""
  semantics = MagicMock(spec=SemanticsManager)

  # Provide an object with model_dump
  class MockAliasInfo:
    def model_dump(self):
      return {"name": "my_alias"}

  semantics.get_framework_config.return_value = {"alias": MockAliasInfo()}

  config = RuntimeConfig(target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)

  assert ctx.alias_map["my_alias"] == "my_alias"

  # Exception case: trigger exception in _hydrate_aliases
  semantics.get_framework_config.return_value = {"alias": {"name": "bad"}}

  class FailingAliasMap(dict):
    def __setitem__(self, key, value):
      raise ValueError("Test Error")

  ctx.alias_map = FailingAliasMap()
  ctx._hydrate_source_aliases()  # Should swallow exception
