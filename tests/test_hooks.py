"""
Tests for Core Hooks API and Metadata Integrity.
Verifies registration mechanism, Context object data structures, and Injection callbacks.
Also tests Data-Driven logic access (PluginTraits, Variants).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from pydantic import BaseModel, ValidationError

from ml_switcheroo.core.hooks import (
  register_hook,
  get_hook,
  HookContext,
  clear_hooks,
  _HOOKS,
)
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import PluginTraits


# Mock object implies we don't need the full logic
class MockSemantics:
  pass


@pytest.fixture(autouse=True)
def clean_registry():
  """Ensure registry is empty before and after each test."""
  clear_hooks()
  yield
  clear_hooks()


def test_hook_context_metadata_isolation():
  """
  Verify that HookContext instances do not share metadata state.
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  ctx1 = HookContext(semantics, config)
  ctx2 = HookContext(semantics, config)

  # Modify ctx1
  ctx1.metadata["scope_id"] = 1
  ctx1.metadata.setdefault("plugin_data", {})["flag"] = True

  # Assert ctx2 is clean
  assert "scope_id" not in ctx2.metadata
  assert "plugin_data" not in ctx2.metadata

  # Assert ctx1 retained data
  assert ctx1.metadata["scope_id"] == 1
  assert ctx1.metadata["plugin_data"]["flag"] is True


def test_hook_context_initialization():
  """
  Verify HookContext stores attributes correctly as documented.
  """
  semantics = MockSemantics()

  config = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    plugin_settings={"any_key": 123},
  )

  ctx = HookContext(semantics, config)

  assert ctx.semantics is semantics
  assert ctx.source_fw == "torch"
  assert ctx.target_fw == "jax"
  assert isinstance(ctx.metadata, dict)
  assert len(ctx.metadata) == 0

  # Test placeholder config method
  assert ctx.config("any_key", default=999) == 123


def test_registration_flow():
  """
  Verify @register_hook adds function to global registry.
  """
  trigger_name = "test_transformation"

  @register_hook(trigger_name)
  def my_transformer(node, _ctx):
    return node

  assert trigger_name in _HOOKS
  assert _HOOKS[trigger_name] == my_transformer
  assert get_hook(trigger_name) == my_transformer


def test_clear_hooks_resets_registry():
  """
  Verify that clear_hooks removes all registered hooks.
  """
  register_hook("temp")((lambda n, c: n))
  assert "temp" in _HOOKS
  clear_hooks()
  assert len(_HOOKS) == 0
  assert get_hook("temp") is None


def test_get_nonexistent_hook():
  """
  Verify get_hook returns None for unknown triggers.
  """
  assert get_hook("unknown_magic") is None


def test_hook_execution_signature():
  """
  Verify the hook is callable with expected arguments (node, ctx).
  """
  trigger_name = "sig_test"

  # 1. Register
  @register_hook(trigger_name)
  def return_new_node(node: cst.Call, _ctx: HookContext):
    # Return a modified node to prove execution
    new_name = cst.Name("visited")
    return node.with_changes(func=new_name)

  # 2. Setup args
  hook = get_hook(trigger_name)
  dummy_node = cst.Call(func=cst.Name("original"))

  # Changed to valid Enums ("torch", "jax")
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  dummy_ctx = HookContext(MockSemantics(), cfg)

  # 3. Execute
  result_node = hook(dummy_node, dummy_ctx)

  # 4. Verify
  assert isinstance(result_node, cst.Call)
  assert result_node.func.value == "visited"


def test_overwrite_hook():
  """
  Verify that registering the same trigger twice overwrites the previous one.
  This allows user plugins to override default plugins.
  """
  trigger = "conflict"

  @register_hook(trigger)
  def hook_a(_node, _ctx):
    return "A"

  assert get_hook(trigger)(None, None) == "A"

  @register_hook(trigger)
  def hook_b(_node, _ctx):
    return "B"

  assert get_hook(trigger)(None, None) == "B"


def test_injection_logic_dispatch():
  """
  Verify `inject_signature_arg` and `inject_preamble` call their respective callbacks.
  """
  # Create Mocks
  mock_arg_injector = MagicMock()
  mock_preamble_injector = MagicMock()

  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  ctx = HookContext(semantics, config, arg_injector=mock_arg_injector, preamble_injector=mock_preamble_injector)

  # 1. Test Argument Injection
  ctx.inject_signature_arg("rng", "jax.Array")
  mock_arg_injector.assert_called_once_with("rng", "jax.Array")

  # 2. Test Preamble Injection
  ctx.inject_preamble("print('hello')")
  mock_preamble_injector.assert_called_once_with("print('hello')")


def test_config_validation_failure():
  """
  Verify `validate_settings` raises Pydantic errors if config doesn't match schema.
  """
  # Global config passes a string where float is expected
  bad_config = RuntimeConfig(plugin_settings={"epsilon": "im_not_a_float"}, strict_mode=False)

  class PluginSchema(BaseModel):
    epsilon: float

  ctx = HookContext(MockSemantics(), bad_config)

  with pytest.raises(ValidationError):
    ctx.validate_settings(PluginSchema)


def test_config_validation_success():
  """
  Verify `validate_settings` returns a populated model.
  """
  good_config = RuntimeConfig(plugin_settings={"epsilon": 0.001, "ignored": "val"}, strict_mode=False)

  class PluginSchema(BaseModel):
    epsilon: float = 1e-5  # Default

  ctx = HookContext(MockSemantics(), good_config)

  model = ctx.validate_settings(PluginSchema)

  assert model.epsilon == 0.001  # Overridden
  assert not hasattr(model, "ignored")  # Filtered


def test_hook_context_traits_access():
  """Verify plugins can access traits."""
  mgr = MagicMock()
  # Mock return value for get_framework_config
  mgr.get_framework_config.return_value = {"plugin_traits": {"requires_explicit_rng": True}}

  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)

  traits = ctx.plugin_traits
  assert isinstance(traits, PluginTraits)
  assert traits.requires_explicit_rng is True
  # Default is False
  assert traits.has_numpy_compatible_arrays is False


def test_hook_context_traits_access_defaults():
  """Verify plugins get defaults if no config/semantics."""
  # Case 1: Semantics missing
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.plugin_traits.requires_explicit_rng is False

  # Case 2: Config missing in Semantics
  mgr = MagicMock()
  mgr.get_framework_config.return_value = {}
  ctx2 = HookContext(mgr, config)
  assert ctx2.plugin_traits.requires_explicit_rng is False


def test_hook_context_variant_lookup():
  """Verify plugins can see current variant config via resolve_variant."""
  mgr = MagicMock()
  # resolve_variant returns a dict
  mgr.resolve_variant.return_value = {"api": "foo", "pack_to_tuple": "axes"}

  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)
  ctx.current_op_id = "Permute"

  var = ctx.current_variant
  assert var is not None
  assert var.api == "foo"
  assert var.pack_to_tuple == "axes"

  # Verify that resolve_variant was called correctly
  mgr.resolve_variant.assert_called_with("Permute", "jax")


def test_hook_context_variant_lookup_missing():
  """Verify variant lookup safely returns None if nothing found."""
  mgr = MagicMock()
  # Returns None
  mgr.resolve_variant.return_value = None

  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)
  ctx.current_op_id = "MissingOp"

  assert ctx.current_variant is None
