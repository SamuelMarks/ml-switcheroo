"""Unit tests for the plugin binding infrastructure and hook context.

This module contains test cases to verify the correctness of the HookContext,
AutoWireSpec, and related plugin registration mechanisms within the ml-switcheroo
core framework.
"""

from ml_switcheroo.core.hooks import AutoWireSpec, HookContext
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import MagicMock


def test_autowirespec():
  """Verifies that AutoWireSpec initializes and stores operation mappings correctly.

  This test instantiates AutoWireSpec with a mock operation mapping and asserts
  that the underlying dictionary behaves as expected.

  Args:
      None

  Returns:
      None
  """
  spec = AutoWireSpec(ops={"test_op": {"api": "test_api"}})
  assert "test_op" in spec.ops
  assert spec.ops["test_op"]["api"] == "test_api"


def test_hookcontext_init():
  """Verifies the proper initialization of the HookContext object.

  Ensures that HookContext correctly receives, stores, and exposes dependencies
  such as SemanticsManager, RuntimeConfig, injectors, and the symbol table,
  along with proper default values for metadata and operation status.

  Args:
      None

  Returns:
      None
  """
  semantics = MagicMock()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  arg_injector = MagicMock()
  preamble_injector = MagicMock()
  symbol_table = MagicMock()

  ctx = HookContext(semantics, config, arg_injector, preamble_injector, symbol_table)
  assert ctx.semantics == semantics
  assert ctx._runtime_config == config
  assert ctx._arg_injector == arg_injector
  assert ctx._preamble_injector == preamble_injector
  assert ctx._symbol_table == symbol_table
  assert ctx.source_fw == "torch"
  assert ctx.target_fw == "jax"
  assert ctx.metadata == {}
  assert ctx.current_op_id is None


def test_hookcontext_resolve_type():
  """Verifies that HookContext.resolve_type resolves node types correctly.

  Tests different return cases of the symbol table type-resolution, including
  mapping to Tensor, Module, other types, and handling of None outcomes.

  Args:
      None

  Returns:
      None
  """
  symbol_table = MagicMock()
  mock_sym = MagicMock()
  mock_sym.name = "Tensor"
  symbol_table.get_type.return_value = mock_sym

  ctx = HookContext(semantics=MagicMock(), config=RuntimeConfig(), symbol_table=symbol_table)

  # Tensor
  assert ctx.resolve_type("node") == "Tensor"

  # Module
  mock_sym.name = "torch.nn.Module"
  assert ctx.resolve_type("node") == "Module"

  # Other
  mock_sym.name = "int"
  assert ctx.resolve_type("node") == "int"

  # None cases
  symbol_table.get_type.return_value = None
  assert ctx.resolve_type("node") is None

  ctx._symbol_table = None
  assert ctx.resolve_type("node") is None


def test_hookcontext_plugin_traits():
  """Verifies retrieval of custom plugin traits within the HookContext.

  Ensures that framework-specific plugin traits can be correctly read
  and parsed from the semantics manager, whether provided as dictionaries,
  PluginTraits objects, or when configurations/semantics are missing.

  Args:
      None

  Returns:
      None
  """
  semantics = MagicMock()

  # Dict traits
  semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx = HookContext(semantics=semantics, config=RuntimeConfig(target_framework="jax"))
  traits = ctx.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # No traits in config
  semantics.get_framework_config.return_value = {}
  traits = ctx.plugin_traits
  assert traits.has_numpy_compatible_arrays is False

  # Object traits
  from ml_switcheroo.semantics.schema import PluginTraits

  semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
  traits = ctx.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # None config
  semantics.get_framework_config.return_value = None
  traits = ctx.plugin_traits
  assert traits.has_numpy_compatible_arrays is False

  # None semantics
  ctx = HookContext(semantics=None, config=RuntimeConfig(target_framework="jax"))
  traits = ctx.plugin_traits
  assert traits.has_numpy_compatible_arrays is False


def test_hookcontext_current_variant():
  """Verifies HookContext.current_variant correctly resolves the current framework variant.

  Tests the integration with the semantics manager's variant resolution
  to fetch the correct FrameworkVariant based on the current operation ID,
  and confirms graceful handling when no variant, operation, or semantics
  manager is set.

  Args:
      None

  Returns:
      None
  """
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "jax.numpy.add", "pack_to_tuple": "True"}

  ctx = HookContext(semantics=semantics, config=RuntimeConfig(target_framework="jax"))
  ctx.current_op_id = "Add"

  variant = ctx.current_variant
  assert variant is not None
  assert variant.api == "jax.numpy.add"
  assert variant.pack_to_tuple == "True"

  # No target variant
  semantics.resolve_variant.return_value = None
  assert ctx.current_variant is None

  # No semantics/current_op
  ctx.current_op_id = None
  assert ctx.current_variant is None
  ctx.current_op_id = "Add"
  ctx.semantics = None
  assert ctx.current_variant is None


def test_hookcontext_injectors():
  """Verifies the behavior of argument and preamble injectors in HookContext.

  Ensures that calling inject_signature_arg or inject_preamble properly
  delegates to their respective callback handlers when configured, and
  does not raise errors when no injector callbacks are registered.

  Args:
      None

  Returns:
      None
  """
  arg_inj = MagicMock()
  preamble_inj = MagicMock()
  ctx = HookContext(semantics=MagicMock(), config=RuntimeConfig(), arg_injector=arg_inj, preamble_injector=preamble_inj)

  ctx.inject_signature_arg("x", "int")
  arg_inj.assert_called_with("x", "int")

  ctx.inject_preamble("print('test')")
  preamble_inj.assert_called_with("print('test')")

  # Should not crash if injectors are None
  ctx = HookContext(semantics=MagicMock(), config=RuntimeConfig(), arg_injector=None, preamble_injector=None)
  ctx.inject_signature_arg("x", "int")
  ctx.inject_preamble("print('test')")


def test_hookcontext_config():
  """Verifies that raw configuration values can be retrieved via HookContext.

  Validates that custom plugin configuration settings can be read by key
  from the underlying runtime configuration, and that requested missing
  keys fall back to their specified default values.

  Args:
      None

  Returns:
      None
  """
  config = RuntimeConfig(plugin_settings={"test_key": "test_val"})
  ctx = HookContext(semantics=MagicMock(), config=config)

  assert ctx.raw_config("test_key") == "test_val"
  assert ctx.raw_config("missing_key", "default") == "default"


def test_hookcontext_validate_settings():
  """Verifies HookContext.validate_settings correctly parses and validates Pydantic schemas.

  Tests the parsing mechanism of custom configuration dictionaries using a Pydantic
  model, confirming that input values are properly typed and extra fields are filtered out.

  Args:
      None

  Returns:
      None
  """
  from pydantic import BaseModel

  class TestSettings(BaseModel):
    """Pydantic model representing mock plugin configuration settings for testing.

    Attributes:
        val1 (str): A required string parameter.
        val2 (int): An optional integer parameter, defaulting to 0.
    """

    val1: str
    val2: int = 0

  config = RuntimeConfig(plugin_settings={"val1": "test", "val2": 1, "extra": True})
  ctx = HookContext(semantics=MagicMock(), config=config)

  validated = ctx.validate_settings(TestSettings)
  assert validated.val1 == "test"
  assert validated.val2 == 1
  assert not hasattr(validated, "extra")


def test_hookcontext_lookup_api():
  """Verifies HookContext.lookup_api retrieves the mapped target API name for an operation.

  Tests that the context queries the semantics manager for the correct variant API
  mapping associated with a given operation identifier, handling cases of missing
  variants or missing semantics.

  Args:
      None

  Returns:
      None
  """
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "jax.numpy.add"}
  ctx = HookContext(semantics=semantics, config=RuntimeConfig(target_framework="jax"))

  assert ctx.lookup_api("Add") == "jax.numpy.add"

  semantics.resolve_variant.return_value = None
  assert ctx.lookup_api("Add") is None

  ctx.semantics = None
  assert ctx.lookup_api("Add") is None


def test_hookcontext_lookup_signature():
  """Verifies HookContext.lookup_signature parses and builds an operation's argument list.

  Tests that the signature parser correctly handles standard argument definitions
  in various configurations (such as standard strings, nested lists, and dictionary
  records), safely returning an empty list if definitions or semantics are missing.

  Args:
      None

  Returns:
      None
  """
  semantics = MagicMock()

  # Test different list types in std_args
  semantics.get_definition_by_id.return_value = {
    "std_args": [
      "arg1",
      ["arg2", "type2"],
      {"name": "arg3", "type": "type3"},
      {"type": "type4"},  # Missing name
    ]
  }

  ctx = HookContext(semantics=semantics, config=RuntimeConfig())
  assert ctx.lookup_signature("Add") == ["arg1", "arg2", "arg3"]

  semantics.get_definition_by_id.return_value = None
  assert ctx.lookup_signature("Add") == []

  ctx.semantics = None
  assert ctx.lookup_signature("Add") == []
