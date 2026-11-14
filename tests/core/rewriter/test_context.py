"""Unit tests for `RewriterContext` (`ml_switcheroo.core.rewriter.context`)."""

from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.exceptions import ConfigurationError
from ml_switcheroo.semantics.manager import SemanticsManager


def test_rewriter_context_rejects_null_semantics() -> None:
  """Verify `RewriterContext` raises `ConfigurationError` when `semantics` is `None`."""
  with pytest.raises(ConfigurationError, match="non-None SemanticsManager"):
    RewriterContext(semantics=None)


def test_rewriter_context_properties_and_default_injectors() -> None:
  """Verify framework properties, `plugin_traits`, argument injection, and preamble deduplication."""
  semantics = SemanticsManager()
  ctx_default = RewriterContext(semantics=semantics, config=None)
  assert ctx_default.config is not None

  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)
  assert ctx.source_framework == "torch"
  assert ctx.target_framework == "jax"
  assert ctx.plugin_traits == ctx.hook_context.plugin_traits

  # Argument injection without active signature stack is a no-op
  ctx._default_arg_injector("device", "str")

  sig_ctx = SignatureContext()
  ctx.signature_stack.append(sig_ctx)
  ctx._default_arg_injector("device", "str")
  ctx._default_arg_injector("device", "int")
  assert sig_ctx.injected_args == [("device", "str")]

  # Preamble injection outside function vs inside function vs import statements
  ctx.signature_stack.pop()
  ctx._default_preamble_injector("x = 1")
  ctx._default_preamble_injector("x = 1")
  assert ctx.module_preamble == ["x = 1"]
  assert "x = 1" in ctx._satisfied_preamble_injections

  ctx.signature_stack.append(sig_ctx)
  ctx._default_preamble_injector("y = 2")
  ctx._default_preamble_injector("y = 2")
  assert sig_ctx.preamble_stmts == ["y = 2"]

  ctx._default_preamble_injector("import os")
  ctx._default_preamble_injector("import os")
  assert "import os" in ctx.module_preamble


def test_hydrate_source_aliases_variants_and_errors() -> None:
  """Verify `_hydrate_source_aliases` handles dicts, `model_dump` objects, empty configs, and exceptions."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  # Dict with `name`
  sm_dict = MagicMock(spec=SemanticsManager)
  sm_dict.get_framework_config.return_value = {"alias": {"name": "th"}}
  assert RewriterContext(semantics=sm_dict, config=config).alias_map == {"th": "th"}

  # Object with `model_dump()`
  class _AliasModelStub:
    """Typed alias config stub providing `model_dump()`."""

    def model_dump(self) -> dict[str, str]:
      """Serialize alias dictionary.

      Returns:
          Alias dictionary with `name`.
      """
      return {"name": "my_alias"}

  sm_model = MagicMock(spec=SemanticsManager)
  sm_model.get_framework_config.return_value = {"alias": _AliasModelStub()}
  assert RewriterContext(semantics=sm_model, config=config).alias_map == {"my_alias": "my_alias"}

  from pydantic import JsonValue

  # Empty dict, missing `name`, non-dict `alias`, `None` config, and raised exceptions
  empty_cfgs: list[dict[str, JsonValue] | None] = [
    {"alias": {}},
    {"alias": {"other": "v"}},
    {"alias": "just_str"},
    None,
  ]
  for cfg_val in empty_cfgs:
    sm_empty = MagicMock(spec=SemanticsManager)
    sm_empty.get_framework_config.return_value = cfg_val
    assert RewriterContext(semantics=sm_empty, config=config).alias_map == {}

  sm_err = MagicMock(spec=SemanticsManager)
  sm_err.get_framework_config.side_effect = KeyError("missing")
  assert RewriterContext(semantics=sm_err, config=config).alias_map == {}
