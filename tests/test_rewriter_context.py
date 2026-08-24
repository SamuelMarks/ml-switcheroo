"""Test module."""

from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


def test_rewriter_context_default_injectors():
  """Test element."""
  semantics = MagicMock(spec=SemanticsManager)
  config = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"
  config.effective_target = "mlx"

  # Mock hydrate aliases
  semantics.get_framework_config.return_value = {"alias": {"name": "torch"}}

  ctx = RewriterContext(semantics=semantics, config=config)

  assert ctx.source_fw == "pytorch"
  assert ctx.target_fw == "mlx"

  # Test arg injector without signature stack
  ctx._default_arg_injector("test_arg", "int")

  # Test arg injector with signature stack
  sig_ctx = SignatureContext(existing_args=set(), injected_args=[], preamble_stmts=[])
  ctx.signature_stack.append(sig_ctx)
  ctx._default_arg_injector("test_arg", "int")
  assert ("test_arg", "int") in sig_ctx.injected_args
  # Duplicate avoid
  ctx._default_arg_injector("test_arg", "float")
  assert len(sig_ctx.injected_args) == 1

  # Test preamble injector without signature stack
  ctx.signature_stack.pop()
  ctx._default_preamble_injector("x = 1")
  assert "x = 1" in ctx.module_preamble
  ctx._default_preamble_injector("x = 1")  # Cache dedup
  assert len(ctx.module_preamble) == 1

  # Test preamble injector with signature stack
  ctx.signature_stack.append(sig_ctx)
  ctx._default_preamble_injector("y = 2")
  assert "y = 2" in sig_ctx.preamble_stmts
  ctx._default_preamble_injector("y = 2")  # Dedup
  assert len(sig_ctx.preamble_stmts) == 1

  # Test preamble injector with signature stack but it's an import
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble


def test_rewriter_context_hydrate_pydantic():
  """Test element."""
  semantics = MagicMock(spec=SemanticsManager)
  config = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  class FakeModel:
    def model_dump(self):
      return {"name": "test_alias"}

  semantics.get_framework_config.return_value = {"alias": FakeModel()}
  ctx = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map["test_alias"] == "test_alias"


def test_rewriter_context_hydrate_exception():
  """Test element."""
  semantics = MagicMock(spec=SemanticsManager)
  config = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  semantics.get_framework_config.side_effect = Exception("test")
  ctx = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map == {}


def test_rewriter_context_hydrate_none():
  """Test element."""
  semantics = MagicMock(spec=SemanticsManager)
  config = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  semantics.get_framework_config.return_value = None
  ctx = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map == {}
