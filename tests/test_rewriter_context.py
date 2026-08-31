"""Test module."""

from typing import Dict
from unittest.mock import MagicMock

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.manager import SemanticsManager


def test_rewriter_context_default_injectors() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock(spec=SemanticsManager)
  config: MagicMock = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"
  config.effective_target = "mlx"

  # Mock hydrate aliases
  semantics.get_framework_config.return_value = {"alias": {"name": "torch"}}

  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)

  assert ctx.source_fw == "pytorch"
  assert ctx.target_fw == "mlx"

  # Test arg injector without signature stack
  ctx._default_arg_injector("test_arg", "int")

  # Test arg injector with signature stack
  sig_ctx: SignatureContext = SignatureContext(existing_args=set(), injected_args=[], preamble_stmts=[])
  ctx.signature_stack.append(sig_ctx)
  ctx._default_arg_injector("test_arg", "int")
  assert ("test_arg", "int") in getattr(sig_ctx, "injected_args")
  # Duplicate avoid
  ctx._default_arg_injector("test_arg", "float")
  assert len(getattr(sig_ctx, "injected_args")) == 1

  # Test preamble injector without signature stack
  ctx.signature_stack.pop()
  ctx._default_preamble_injector("x = 1")
  assert "x = 1" in ctx.module_preamble
  ctx._default_preamble_injector("x = 1")  # Cache dedup
  assert len(ctx.module_preamble) == 1

  # Test preamble injector with signature stack
  ctx.signature_stack.append(sig_ctx)
  ctx._default_preamble_injector("y = 2")
  assert "y = 2" in getattr(sig_ctx, "preamble_stmts")
  ctx._default_preamble_injector("y = 2")  # Dedup
  assert len(getattr(sig_ctx, "preamble_stmts")) == 1

  # Test preamble injector with signature stack but it's an import
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble


def test_rewriter_context_hydrate_pydantic() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock(spec=SemanticsManager)
  config: MagicMock = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  class FakeModel:
    """Docstring."""

    def model_dump(self) -> Dict[str, str]:
      """Docstring."""
      return {"name": "test_alias"}

  semantics.get_framework_config.return_value = {"alias": FakeModel()}
  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map["test_alias"] == "test_alias"


def test_rewriter_context_hydrate_exception() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock(spec=SemanticsManager)
  config: MagicMock = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  semantics.get_framework_config.side_effect = Exception("test")
  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map == {}


def test_rewriter_context_hydrate_none() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock(spec=SemanticsManager)
  config: MagicMock = MagicMock(spec=RuntimeConfig)
  config.effective_source = "pytorch"

  semantics.get_framework_config.return_value = None
  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)
  assert ctx.alias_map == {}


# --- Merged from test_rewriter_context_extra.py ---


def test_hydrate_alias_map_with_dict() -> None:
  """Docstring."""
  # Hit the line 160->exit path
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"name": "th"}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)

  assert ctx.alias_map.get("th") == "th"


def test_hydrate_alias_map_no_name() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert "th" not in ctx.alias_map


def test_hydrate_alias_map_model_dump() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()

  class DummyAlias:
    """Docstring."""

    def model_dump(self) -> Dict[str, str]:
      """Docstring."""
      return {"name": "th_dummy"}

  sm.get_framework_config.return_value = {"alias": DummyAlias()}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert ctx.alias_map.get("th_dummy") == "th_dummy"


def test_hydrate_alias_map_dict_no_name_key() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": {"something_else": "value"}}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map


def test_hydrate_alias_map_not_dict() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  sm: MagicMock = MagicMock()
  sm.get_framework_config.return_value = {"alias": "just_a_string"}
  ctx: RewriterContext = RewriterContext(semantics=sm, config=config)
  assert not ctx.alias_map
