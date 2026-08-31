"""Test suite for the Hooks Context module."""

from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, PluginTraits
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Docstring."""
  mgr: MagicMock = MagicMock(spec=SemanticsManager)
  data: Dict[str, Dict[str, Union[List[Union[str, tuple[str, str]]], Dict[str, Dict[str, str]]]]] = {
    "add": {"std_args": ["x1", "x2"], "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}}},
    "abs": {"std_args": [("x", "Array")], "variants": {}},
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  def resolve(aid: str, fw: str) -> Optional[Dict[str, str]]:
    """Resolves ."""
    if aid in data and "variants" in data[aid] and fw in data[aid]["variants"]:
      return data[aid]["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)
  mgr.get_framework_config.return_value = {}
  return mgr


def test_lookup_api_success(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup API successfully."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  result: Optional[str] = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup API different target."""
  config: RuntimeConfig = RuntimeConfig(target_framework="numpy")
  ctx: HookContext = HookContext(mock_semantics, config)
  result: Optional[str] = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup API missing variant."""
  config: RuntimeConfig = RuntimeConfig(target_framework="tensorflow")
  ctx: HookContext = HookContext(mock_semantics, config)
  result: Optional[str] = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup API missing op."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  result: Optional[str] = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup API plugin variant."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  result: Optional[str] = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup signature standard list."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  sig: List[str] = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup signature typed tuples."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  sig: List[str] = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of lookup signature unknown returns empty."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  sig: List[str] = ctx.lookup_signature("ghost_op")
  assert sig == []


def test_hooks_resolve_type_no_symbol_table() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)  # MagicMock instead of None because it expects SemanticsManager
  assert ctx.resolve_type(None) is None


def test_hooks_resolve_type_with_symbol_table() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)

  class DummySym:
    """Docstring."""

    def __init__(self, name: str) -> None:
      """Init."""
      self.name: str = name

  class MockSymbolTable:
    """Docstring."""

    def get_type(self, node: str) -> Optional[DummySym]:
      """Get type."""
      if node == "tensor":
        return DummySym("SomeTensorType")
      elif node == "module":
        return DummySym("SomeModuleType")
      elif node == "other":
        return DummySym("OtherType")
      return None

  ctx._symbol_table = MockSymbolTable()

  assert ctx.resolve_type("none") is None
  assert ctx.resolve_type("tensor") == "Tensor"
  assert ctx.resolve_type("module") == "Module"
  assert ctx.resolve_type("other") == "OtherType"


def test_hooks_plugin_traits_no_semantics() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)
  ctx.semantics.get_framework_config.return_value = {}
  traits: PluginTraits = ctx.plugin_traits
  assert traits is not None
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_with_dict(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {"some_trait": True}}

  with __import__("unittest.mock").mock.patch("ml_switcheroo.core.hooks.PluginTraits.model_validate") as mock_validate:
    mock_validate.return_value = "validated_traits"
    traits = ctx.plugin_traits
    assert traits == "validated_traits"


def test_hooks_plugin_traits_with_object(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  from ml_switcheroo.core.hooks import PluginTraits

  ctx: HookContext = HookContext(mock_semantics, config)
  pt: PluginTraits = PluginTraits()
  mock_semantics.get_framework_config.return_value = {"plugin_traits": pt}
  traits: PluginTraits = ctx.plugin_traits
  assert traits is pt


def test_hooks_plugin_traits_with_other(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": "unsupported"}
  traits: PluginTraits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_falsy(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  # The 'plugin_traits' key exists but value is falsy, e.g., empty dict or None
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {}}
  traits: PluginTraits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_current_variant_no_semantics() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_no_op_id(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_not_resolved(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  ctx.current_op_id = "missing"
  mock_semantics.resolve_variant.return_value = None
  assert ctx.current_variant is None


def test_hooks_current_variant_resolved(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  ctx.current_op_id = "add"

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.hooks.FrameworkVariant.model_validate"
  ) as mock_validate:
    mock_validate.return_value = "validated_variant"
    var = ctx.current_variant
    assert var == "validated_variant"


def test_hooks_inject_signature_arg() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)
  # no injector, should not crash
  ctx.inject_signature_arg("x", "int")

  calls: List[tuple[str, str]] = []

  def mock_injector(name: str, ann: str) -> None:
    """Mock injector."""
    calls.append((name, ann))

  ctx._arg_injector = mock_injector
  ctx.inject_signature_arg("y", "float")
  assert calls == [("y", "float")]


def test_hooks_lookup_api_no_semantics() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  mock_manager = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: HookContext = HookContext(mock_manager, config)
  assert ctx.lookup_signature("add") == []


def test_hooks_lookup_signature_with_dict(mock_semantics: MagicMock) -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(target_framework="jax")
  ctx: HookContext = HookContext(mock_semantics, config)
  mock_semantics.get_definition_by_id.side_effect = lambda aid: {
    "std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]
  }
  mock_semantics.get_definition_by_id.return_value = {"std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]}
  sig: List[str] = ctx.lookup_signature("any")
  assert sig == ["dict_arg", "string_arg"]
