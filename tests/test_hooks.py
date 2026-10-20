"""Test suite for the Hooks module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import BaseModel, ValidationError

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import _HOOKS, HookContext, get_hook, register_hook
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  pass


@pytest.fixture(autouse=True)
def clean_registry() -> typing.Generator[None, None, None]:
  """Helper to clean registry.

  Returns:
      Computed test result or fixture value.
  """
  pass
  yield
  pass


def test_hook_context_metadata_isolation() -> None:
  """Verifies the behavior of hook context metadata isolation."""
  semantics: MockSemantics = MockSemantics()
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx1: HookContext = HookContext(semantics, config)
  ctx2: HookContext = HookContext(semantics, config)
  ctx1.metadata["scope_id"] = 1
  ctx1.metadata.setdefault("plugin_data", {})["flag"] = True  # type: ignore[index, call-overload]
  assert "scope_id" not in ctx2.metadata
  assert "plugin_data" not in ctx2.metadata
  assert ctx1.metadata["scope_id"] == 1
  assert ctx1.metadata["plugin_data"]["flag"] is True  # type: ignore[call-overload, index]


def test_hook_context_initialization() -> None:
  """Verifies the behavior of hook context initialization."""
  semantics: MockSemantics = MockSemantics()
  config: RuntimeConfig = RuntimeConfig(
    source_framework="torch", target_framework="jax", plugin_settings={"any_key": 123}
  )
  ctx: HookContext = HookContext(semantics, config)
  assert ctx.semantics is semantics
  assert ctx.source_fw == "torch"
  assert ctx.target_fw == "jax"
  assert isinstance(ctx.metadata, dict)
  assert len(ctx.metadata) == 0
  assert ctx.raw_config("any_key", default=999) == 123


def test_registration_flow() -> None:
  """Verifies the behavior of registration flow."""
  trigger_name: str = "test_transformation"

  @register_hook(trigger_name)  # type: ignore[type-var]
  def my_transformer(node: cst.CSTNode, _ctx: HookContext) -> cst.CSTNode:
    """Helper to my transformer.

    Args:
        node: Test argument or fixture.
        _ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return node

  assert trigger_name in _HOOKS
  assert _HOOKS[trigger_name] == my_transformer
  assert get_hook(trigger_name) == my_transformer


def test_clear_hooks_resets_registry() -> None:
  """Verifies the behavior of clear hooks resets registry."""
  from unittest.mock import patch

  with patch.dict("ml_switcheroo.core.hooks_registry._HOOKS", {}, clear=True):

    @register_hook("temp")  # type: ignore[type-var]
    def temp_hook(n: cst.CSTNode, c: HookContext) -> cst.CSTNode:
      """Execute temp hook helper logic.

      Args:
          n: Test argument or fixture.
          c: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return n

    assert "temp" in _HOOKS
    _HOOKS.clear()
    assert len(_HOOKS) == 0
  assert get_hook("temp") is None


def test_get_nonexistent_hook() -> None:
  """Gets nonexistent hook."""
  assert get_hook("unknown_magic") is None


def test_hook_execution_signature() -> None:
  """Verifies the behavior of hook execution signature."""
  trigger_name: str = "sig_test"

  @register_hook(trigger_name)  # type: ignore[type-var]
  def return_new_node(node: cst.Call, _ctx: HookContext) -> cst.Call:
    """Helper to return new node.

    Args:
        node: Test argument or fixture.
        _ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    new_name: cst.Name = cst.Name("visited")
    return node.with_changes(func=new_name)

  hook = get_hook(trigger_name)
  assert hook is not None
  dummy_node: cst.Call = cst.Call(func=cst.Name("original"))
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  dummy_ctx: HookContext = HookContext(MockSemantics(), cfg)
  result_node: cst.Call = hook(dummy_node, dummy_ctx)  # type: ignore[assignment]
  assert isinstance(result_node, cst.Call)
  assert getattr(result_node.func, "value", None) == "visited"


def test_overwrite_hook() -> None:
  """Verifies the behavior of overwrite hook."""
  trigger: str = "conflict"

  @register_hook(trigger)  # type: ignore[type-var]
  def hook_a(_node: typing.Optional[cst.CSTNode], _ctx: typing.Optional[HookContext]) -> str:
    """Helper to hook a.

    Args:
        _node: Test argument or fixture.
        _ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return "A"

  hook_a_res = get_hook(trigger)
  assert hook_a_res is not None
  assert hook_a_res(None, None) == "A"  # type: ignore[arg-type]

  @register_hook(trigger)  # type: ignore[type-var]
  def hook_b(_node: typing.Optional[cst.CSTNode], _ctx: typing.Optional[HookContext]) -> str:
    """Helper to hook b.

    Args:
        _node: Test argument or fixture.
        _ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return "B"

  hook_b_res = get_hook(trigger)
  assert hook_b_res is not None
  assert hook_b_res(None, None) == "B"  # type: ignore[arg-type]


def test_injection_logic_dispatch() -> None:
  """Verifies the behavior of injection logic dispatch."""
  mock_arg_injector: MagicMock = MagicMock()
  mock_preamble_injector: MagicMock = MagicMock()
  semantics: MockSemantics = MockSemantics()
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx: HookContext = HookContext(
    semantics, config, arg_injector=mock_arg_injector, preamble_injector=mock_preamble_injector
  )
  ctx.inject_signature_arg("rng", "jax.Array")
  mock_arg_injector.assert_called_once_with("rng", "jax.Array")
  ctx.inject_preamble("print('hello')")
  mock_preamble_injector.assert_called_once_with("print('hello')")


def test_config_validation_failure() -> None:
  """Verifies the behavior of configuration validation successfully handling failure."""
  bad_config: RuntimeConfig = RuntimeConfig(plugin_settings={"epsilon": "im_not_a_float"}, strict_mode=False)

  class PluginSchema(BaseModel):
    """Test helper class PluginSchema."""

    epsilon: float

  ctx: HookContext = HookContext(MockSemantics(), bad_config)
  with pytest.raises(ValidationError):
    ctx.validate_settings(PluginSchema)


def test_config_validation_success() -> None:
  """Verifies the behavior of configuration validation successfully."""
  good_config: RuntimeConfig = RuntimeConfig(plugin_settings={"epsilon": 0.001, "ignored": "val"}, strict_mode=False)

  class PluginSchema(BaseModel):
    """Test helper class PluginSchema."""

    epsilon: float = 1e-05

  ctx: HookContext = HookContext(MockSemantics(), good_config)
  model = ctx.validate_settings(PluginSchema)
  assert model.epsilon == 0.001
  assert not hasattr(model, "ignored")


def test_hook_context_traits_access() -> None:
  """Verifies the behavior of hook context traits access."""
  mgr: MagicMock = MagicMock()
  mgr.get_framework_config.return_value = {"plugin_traits": {"requires_explicit_rng": True}}
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mgr, config)
  traits: PluginTraits = ctx.plugin_traits
  assert isinstance(traits, PluginTraits)
  assert traits.requires_explicit_rng is True
  assert traits.has_numpy_compatible_arrays is False


def test_hook_context_traits_access_defaults() -> None:
  """Verifies the behavior of hook context traits access defaults."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(MagicMock(), config)  # Can't use None if type requires SemanticsManager
  ctx.semantics.get_framework_config.return_value = {}  # type: ignore[union-attr]  # Mock it properly
  assert ctx.plugin_traits.requires_explicit_rng is False
  mgr: MagicMock = MagicMock()
  mgr.get_framework_config.return_value = {}
  ctx2: HookContext = HookContext(mgr, config)
  assert ctx2.plugin_traits.requires_explicit_rng is False


def test_hook_context_variant_lookup() -> None:
  """Verifies the behavior of hook context variant lookup."""
  mgr: MagicMock = MagicMock()
  mgr.resolve_variant.return_value = {"api": "foo", "pack_to_tuple": "axes"}
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mgr, config)
  ctx.current_op_id = "Permute"
  var = ctx.current_variant
  assert var is not None
  assert var.api == "foo"
  assert var.pack_to_tuple == "axes"
  mgr.resolve_variant.assert_called_with("Permute", "jax")


def test_hook_context_variant_lookup_missing() -> None:
  """Verifies the behavior of hook context variant lookup missing."""
  mgr: MagicMock = MagicMock()
  mgr.resolve_variant.return_value = None
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mgr, config)
  ctx.current_op_id = "MissingOp"
  assert ctx.current_variant is None


def test_hooks_module_plugins_loaded_sync() -> None:
  """Verifies that hooks._PLUGINS_LOADED synchronizes with hooks_registry."""
  import ml_switcheroo.core.hooks as h
  import ml_switcheroo.core.hooks_registry as hr

  h._PLUGINS_LOADED = True
  assert hr._PLUGINS_LOADED is True
  assert h._PLUGINS_LOADED is True

  h._PLUGINS_LOADED = False
  assert hr._PLUGINS_LOADED is False
  assert h._PLUGINS_LOADED is False  # type: ignore[unreachable]
