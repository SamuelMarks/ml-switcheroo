"""Test suite for the Hooks Context module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  data = {
    "add": {"std_args": ["x1", "x2"], "variants": {"jax": {"api": "jax.numpy.add"}, "numpy": {"api": "numpy.add"}}},
    "abs": {"std_args": [("x", "Array")], "variants": {}},
    "complex": {"variants": {"jax": {"requires_plugin": "magic"}}},
  }

  def resolve(aid, fw):
    """Resolves ."""
    if aid in data and fw in data[aid]["variants"]:
      return data[aid]["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition_by_id.side_effect = lambda aid: data.get(aid)
  mgr.get_framework_config.return_value = {}
  return mgr


def test_lookup_api_success(mock_semantics):
  """Verifies the behavior of lookup API successfully."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "jax.numpy.add"


def test_lookup_api_different_target(mock_semantics):
  """Verifies the behavior of lookup API different target."""
  config = RuntimeConfig(target_framework="numpy")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result == "numpy.add"


def test_lookup_api_missing_variant(mock_semantics):
  """Verifies the behavior of lookup API missing variant."""
  config = RuntimeConfig(target_framework="tensorflow")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("add")
  assert result is None


def test_lookup_api_missing_op(mock_semantics):
  """Verifies the behavior of lookup API missing op."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("unknown_logic")
  assert result is None


def test_lookup_api_plugin_variant(mock_semantics):
  """Verifies the behavior of lookup API plugin variant."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  result = ctx.lookup_api("complex")
  assert result is None


def test_lookup_signature_standard_list(mock_semantics):
  """Verifies the behavior of lookup signature standard list."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("add")
  assert sig == ["x1", "x2"]


def test_lookup_signature_typed_tuples(mock_semantics):
  """Verifies the behavior of lookup signature typed tuples."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("abs")
  assert sig == ["x"]


def test_lookup_signature_unknown_returns_empty(mock_semantics):
  """Verifies the behavior of lookup signature unknown returns empty."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  sig = ctx.lookup_signature("ghost_op")
  assert sig == []


def test_hooks_resolve_type_no_symbol_table():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.resolve_type(None) is None


def test_hooks_resolve_type_with_symbol_table():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)

  class DummySym:
    """Test class."""

    def __init__(self, name):
      self.name = name

  class MockSymbolTable:
    """Test class."""

    def get_type(self, node):
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


def test_hooks_plugin_traits_no_semantics():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  traits = ctx.plugin_traits
  assert traits is not None
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_with_dict(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {"some_trait": True}}

  with __import__("unittest.mock").mock.patch("ml_switcheroo.core.hooks.PluginTraits.model_validate") as mock_validate:
    mock_validate.return_value = "validated_traits"
    traits = ctx.plugin_traits
    assert traits == "validated_traits"


def test_hooks_plugin_traits_with_object(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  from ml_switcheroo.core.hooks import PluginTraits

  ctx = HookContext(mock_semantics, config)
  pt = PluginTraits()
  mock_semantics.get_framework_config.return_value = {"plugin_traits": pt}
  traits = ctx.plugin_traits
  assert traits is pt


def test_hooks_plugin_traits_with_other(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  mock_semantics.get_framework_config.return_value = {"plugin_traits": "unsupported"}
  traits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_plugin_traits_falsy(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  # The 'plugin_traits' key exists but value is falsy, e.g., empty dict or None
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {}}
  traits = ctx.plugin_traits
  assert type(traits).__name__ == "PluginTraits"


def test_hooks_current_variant_no_semantics():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_no_op_id(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  assert ctx.current_variant is None


def test_hooks_current_variant_not_resolved(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  ctx.current_op_id = "missing"
  mock_semantics.resolve_variant.return_value = None
  assert ctx.current_variant is None


def test_hooks_current_variant_resolved(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  ctx.current_op_id = "add"

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.hooks.FrameworkVariant.model_validate"
  ) as mock_validate:
    mock_validate.return_value = "validated_variant"
    var = ctx.current_variant
    assert var == "validated_variant"


def test_hooks_inject_signature_arg():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  # no injector, should not crash
  ctx.inject_signature_arg("x", "int")

  calls = []

  def mock_injector(name, ann):
    calls.append((name, ann))

  ctx._arg_injector = mock_injector
  ctx.inject_signature_arg("y", "float")
  assert calls == [("y", "float")]


def test_hooks_lookup_api_no_semantics():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.lookup_api("add") is None


def test_hooks_lookup_signature_no_semantics():
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(None, config)
  assert ctx.lookup_signature("add") == []


def test_hooks_lookup_signature_with_dict(mock_semantics):
  """Test method."""
  config = RuntimeConfig(target_framework="jax")
  ctx = HookContext(mock_semantics, config)
  mock_semantics.get_definition_by_id.side_effect = None
  mock_semantics.get_definition_by_id.return_value = {"std_args": [{"name": "dict_arg"}, {"no_name": True}, "string_arg"]}
  sig = ctx.lookup_signature("any")
  assert sig == ["dict_arg", "string_arg"]
