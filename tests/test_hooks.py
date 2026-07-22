"""Test suite for the Hooks module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from pydantic import BaseModel, ValidationError
from ml_switcheroo.core.hooks import register_hook, get_hook, HookContext, _HOOKS
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import PluginTraits


class MockSemantics:
  """Mock Semantics class for testing purposes."""

  pass


@pytest.fixture(autouse=True)
def clean_registry():
  """Helper to clean registry."""
  pass
  yield
  pass


def test_hook_context_metadata_isolation():
  """Verifies the behavior of hook context metadata isolation."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx1 = HookContext(semantics, config)
  ctx2 = HookContext(semantics, config)
  ctx1.metadata["scope_id"] = 1
  ctx1.metadata.setdefault("plugin_data", {})["flag"] = True
  assert "scope_id" not in ctx2.metadata
  assert "plugin_data" not in ctx2.metadata
  assert ctx1.metadata["scope_id"] == 1
  assert ctx1.metadata["plugin_data"]["flag"] is True


def test_hook_context_initialization():
  """Verifies the behavior of hook context initialization."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", plugin_settings={"any_key": 123})
  ctx = HookContext(semantics, config)
  assert ctx.semantics is semantics
  assert ctx.source_fw == "torch"
  assert ctx.target_fw == "jax"
  assert isinstance(ctx.metadata, dict)
  assert len(ctx.metadata) == 0
  assert ctx.raw_config("any_key", default=999) == 123


def test_registration_flow():
  """Verifies the behavior of registration flow."""
  trigger_name = "test_transformation"

  @register_hook(trigger_name)
  def my_transformer(node, _ctx):
    """Helper to my transformer."""
    return node

  assert trigger_name in _HOOKS
  assert _HOOKS[trigger_name] == my_transformer
  assert get_hook(trigger_name) == my_transformer


def test_clear_hooks_resets_registry():
  """Verifies the behavior of clear hooks resets registry."""
  from unittest.mock import patch

  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):
    register_hook("temp")(lambda n, c: n)
    assert "temp" in _HOOKS
    _HOOKS.clear()
    assert len(_HOOKS) == 0
  assert get_hook("temp") is None


def test_get_nonexistent_hook():
  """Gets nonexistent hook."""
  assert get_hook("unknown_magic") is None


def test_hook_execution_signature():
  """Verifies the behavior of hook execution signature."""
  trigger_name = "sig_test"

  @register_hook(trigger_name)
  def return_new_node(node: cst.Call, _ctx: HookContext):
    """Helper to return new node."""
    new_name = cst.Name("visited")
    return node.with_changes(func=new_name)

  hook = get_hook(trigger_name)
  dummy_node = cst.Call(func=cst.Name("original"))
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  dummy_ctx = HookContext(MockSemantics(), cfg)
  result_node = hook(dummy_node, dummy_ctx)
  assert isinstance(result_node, cst.Call)
  assert result_node.func.value == "visited"


def test_overwrite_hook():
  """Verifies the behavior of overwrite hook."""
  trigger = "conflict"

  @register_hook(trigger)
  def hook_a(_node, _ctx):
    """Helper to hook a."""
    return "A"

  assert get_hook(trigger)(None, None) == "A"

  @register_hook(trigger)
  def hook_b(_node, _ctx):
    """Helper to hook b."""
    return "B"

  assert get_hook(trigger)(None, None) == "B"


def test_injection_logic_dispatch():
  """Verifies the behavior of injection logic dispatch."""
  mock_arg_injector = MagicMock()
  mock_preamble_injector = MagicMock()
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = HookContext(semantics, config, arg_injector=mock_arg_injector, preamble_injector=mock_preamble_injector)
  ctx.inject_signature_arg("rng", "jax.Array")
  mock_arg_injector.assert_called_once_with("rng", "jax.Array")
  ctx.inject_preamble("print('hello')")
  mock_preamble_injector.assert_called_once_with("print('hello')")


def test_config_validation_failure():
  """Verifies the behavior of configuration validation successfully handling failure."""
  bad_config = RuntimeConfig(plugin_settings={"epsilon": "im_not_a_float"}, strict_mode=False)

  class PluginSchema(BaseModel):
    """Test suite for the Plugin Schema component."""

    epsilon: float

  ctx = HookContext(MockSemantics(), bad_config)
  with pytest.raises(ValidationError):
    ctx.validate_settings(PluginSchema)


def test_config_validation_success():
  """Verifies the behavior of configuration validation successfully."""
  good_config = RuntimeConfig(plugin_settings={"epsilon": 0.001, "ignored": "val"}, strict_mode=False)

  class PluginSchema(BaseModel):
    """Test suite for the Plugin Schema component."""

    epsilon: float = 1e-05

  ctx = HookContext(MockSemantics(), good_config)
  model = ctx.validate_settings(PluginSchema)
  assert model.epsilon == 0.001
  assert not hasattr(model, "ignored")


def test_hook_context_traits_access():
  """Verifies the behavior of hook context traits access."""
  mgr = MagicMock()
  mgr.get_framework_config.return_value = {"plugin_traits": {"requires_explicit_rng": True}}
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)
  traits = ctx.plugin_traits
  assert isinstance(traits, PluginTraits)
  assert traits.requires_explicit_rng is True
  assert traits.has_numpy_compatible_arrays is False


def test_hook_context_traits_access_defaults():
  """Verifies the behavior of hook context traits access defaults."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.plugin_traits.requires_explicit_rng is False
  mgr = MagicMock()
  mgr.get_framework_config.return_value = {}
  ctx2 = HookContext(mgr, config)
  assert ctx2.plugin_traits.requires_explicit_rng is False


def test_hook_context_variant_lookup():
  """Verifies the behavior of hook context variant lookup."""
  mgr = MagicMock()
  mgr.resolve_variant.return_value = {"api": "foo", "pack_to_tuple": "axes"}
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)
  ctx.current_op_id = "Permute"
  var = ctx.current_variant
  assert var is not None
  assert var.api == "foo"
  assert var.pack_to_tuple == "axes"
  mgr.resolve_variant.assert_called_with("Permute", "jax")


def test_hook_context_variant_lookup_missing():
  """Verifies the behavior of hook context variant lookup missing."""
  mgr = MagicMock()
  mgr.resolve_variant.return_value = None
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mgr, config)
  ctx.current_op_id = "MissingOp"
  assert ctx.current_variant is None
