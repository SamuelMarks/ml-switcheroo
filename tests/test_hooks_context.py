"""Test suite for the Hooks Context module."""

import typing

from unittest.mock import MagicMock

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, PluginTraits
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics() -> typing.Any:
  """Execute mock semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: typing.Any = MagicMock(spec=SemanticsManager)
  data: typing.Any = {
    "add": {"std_args": ["x1", "x2"], "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}}},
    "abs": {"std_args": [("x", "Array")], "variants": {}},
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  def resolve(aid: typing.Any, fw: typing.Any) -> typing.Any:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid in data and "variants" in data[aid] and fw in data[aid]["variants"]:
      return data[aid]["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)
  mgr.get_framework_config.return_value = {}
  return mgr


def test_lookup_api_success(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup API successfully.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  result: typing.Any = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup API different target.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="numpy")
  ctx: typing.Any = HookContext(mock_semantics, config)
  result: typing.Any = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup API missing variant.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="tensorflow")
  ctx: typing.Any = HookContext(mock_semantics, config)
  result: typing.Any = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup API missing op.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  result: typing.Any = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup API plugin variant.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  result: typing.Any = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup signature standard list.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  sig: typing.Any = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup signature typed tuples.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  sig: typing.Any = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics: typing.Any) -> None:
  """Verifies the behavior of lookup signature unknown returns empty.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  sig: typing.Any = ctx.lookup_signature("ghost_op")
  assert sig == []


def test_hooks_resolve_type_no_symbol_table() -> None:
  """Verify hooks resolve type no symbol table behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)  # MagicMock instead of None because it expects SemanticsManager
  assert ctx.resolve_type(None) is None


def test_hooks_resolve_type_with_symbol_table() -> None:
  """Verify hooks resolve type with symbol table behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)

  class DummySym:
    """Test helper class DummySym."""

    def __init__(self, name: typing.Any) -> None:
      """Init.

      Args:
          name: Test argument or fixture.
      """
      self.name: typing.Any = name

  class MockSymbolTable:
    """Test helper class MockSymbolTable."""

    def get_type(self, node: typing.Any) -> typing.Any:
      """Get type.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
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
  """Verify hooks plugin traits no semantics behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)
  ctx.semantics.get_framework_config.return_value = {}
  traits: typing.Any = ctx.plugin_traits
  assert traits is not None
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_with_dict(mock_semantics: typing.Any) -> None:
  """Verify hooks plugin traits with dict behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {"some_trait": True}}

  with __import__("unittest.mock").mock.patch("ml_switcheroo.core.hooks.PluginTraits.model_validate") as mock_validate:
    mock_validate.return_value = "validated_traits"
    traits: typing.Any = ctx.plugin_traits
    assert traits == "validated_traits"


def test_hooks_plugin_traits_with_object(mock_semantics: typing.Any) -> None:
  """Verify hooks plugin traits with object behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  from ml_switcheroo.core.hooks import PluginTraits

  ctx: typing.Any = HookContext(mock_semantics, config)
  pt: typing.Any = PluginTraits()
  mock_semantics.get_framework_config.return_value = {"plugin_traits": pt}
  traits: typing.Any = ctx.plugin_traits
  assert traits is pt


def test_hooks_plugin_traits_with_other(mock_semantics: typing.Any) -> None:
  """Verify hooks plugin traits with other behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": "unsupported"}
  traits: typing.Any = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_falsy(mock_semantics: typing.Any) -> None:
  """Verify hooks plugin traits falsy behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  # The 'plugin_traits' key exists but value is falsy, e.g., empty dict or None
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {}}
  traits: typing.Any = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_current_variant_no_semantics() -> None:
  """Verify hooks current variant no semantics behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_no_op_id(mock_semantics: typing.Any) -> None:
  """Verify hooks current variant no op id behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_not_resolved(mock_semantics: typing.Any) -> None:
  """Verify hooks current variant not resolved behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  ctx.current_op_id = "missing"
  mock_semantics.resolve_variant.return_value = None
  assert ctx.current_variant is None


def test_hooks_current_variant_resolved(mock_semantics: typing.Any) -> None:
  """Verify hooks current variant resolved behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  ctx.current_op_id = "add"

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.hooks.FrameworkVariant.model_validate"
  ) as mock_validate:
    mock_validate.return_value = "validated_variant"
    var: typing.Any = ctx.current_variant
    assert var == "validated_variant"


def test_hooks_inject_signature_arg() -> None:
  """Verify hooks inject signature arg behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)
  # no injector, should not crash
  ctx.inject_signature_arg("x", "int")

  calls: typing.Any = []

  def mock_injector(name: typing.Any, ann: typing.Any) -> None:
    """Mock injector.

    Args:
        name: Test argument or fixture.
        ann: Test argument or fixture.
    """
    calls.append((name, ann))

  ctx._arg_injector = mock_injector
  ctx.inject_signature_arg("y", "float")
  assert calls == [("y", "float")]


def test_hooks_lookup_api_no_semantics() -> None:
  """Verify hooks lookup api no semantics behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics() -> None:
  """Verify hooks lookup signature no semantics behavior."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  mock_manager: typing.Any = MagicMock()
  mock_manager.resolve_variant.return_value = None
  ctx: typing.Any = HookContext(mock_manager, config)
  assert ctx.lookup_signature("add") == []


def test_hooks_lookup_signature_with_dict(mock_semantics: typing.Any) -> None:
  """Verify hooks lookup signature with dict behavior.

  Args:
      mock_semantics: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(mock_semantics, config)
  mock_semantics.get_definition_by_id.side_effect = lambda aid: {
    "std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]
  }
  mock_semantics.get_definition_by_id.return_value = {"std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]}
  sig: typing.Any = ctx.lookup_signature("any")
  assert sig == ["dict_arg", "string_arg"]


def test_hooks_plugin_traits_branches() -> None:
  """Verifies plugin_traits returns default PluginTraits when semantics is None or config is None."""
  config: typing.Any = RuntimeConfig(target_framework="jax")

  # Branch 1: semantics is None
  ctx_no_sem: typing.Any = HookContext(None, config)
  traits1: typing.Any = ctx_no_sem.plugin_traits
  assert isinstance(traits1, PluginTraits)

  # Branch 2: framework config is None
  mock_semantics: typing.Any = MagicMock(spec=SemanticsManager)
  mock_semantics.get_framework_config.return_value = None
  ctx_no_conf: typing.Any = HookContext(mock_semantics, config)
  traits2: typing.Any = ctx_no_conf.plugin_traits
  assert isinstance(traits2, PluginTraits)
  assert traits2.has_numpy_compatible_arrays is False


def test_hooks_inject_preamble_branches() -> None:
  """Verifies inject_preamble handles both present and missing preamble injectors."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(None, config)
  # Branch 1: None
  ctx.inject_preamble("import math")

  # Branch 2: Present injector
  injected: typing.Any = []
  ctx._preamble_injector = lambda s: injected.append(s)
  ctx.inject_preamble("import os")
  assert injected == ["import os"]


def test_hooks_lookup_api_no_semantics_explicit() -> None:
  """Verifies lookup_api returns None when semantics is None."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(None, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics_explicit() -> None:
  """Verifies lookup_signature returns empty list when semantics is None."""
  config: typing.Any = RuntimeConfig(target_framework="jax")
  ctx: typing.Any = HookContext(None, config)
  assert ctx.lookup_signature("add") == []
