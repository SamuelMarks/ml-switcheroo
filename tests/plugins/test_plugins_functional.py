"""Test suite for the Plugins module."""

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.frameworks.base import get_adapter, register_framework
from ml_switcheroo.semantics.manager import SemanticsManager


def cleanup_args(args_list: list[cst.Arg]) -> list[cst.Arg]:
  """Execute `with_changes` to support cleanup args in plugins tests.

  Args:
      args_list: Input `args_list` parameter value (`list[cst.Arg]`).

  Returns:
      Resulting `list[cst.Arg]` from cleanup args.
  """
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` used in plugins tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, dict[str, JsonValue]] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.framework_configs: dict[str, dict[str, JsonValue]] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, bool] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, dict[SemanticTier | str, dict[str, str | None]]] = {}
    self._source_registry: dict[str, tuple[str, SemanticTier | str]] = {}
    special_def: dict[str, JsonValue] = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    self.data["special_add"] = special_def
    self._reverse_index["torch.special_add"] = ("special_add", special_def)
    add_def: dict[str, JsonValue] = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self.data["add"] = add_def
    self._reverse_index["jax.numpy.add"] = ("add", add_def)
    self._reverse_index["torch.add"] = ("add", add_def)

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Resulting `set[str]` from get all rng methods.
    """
    return self._known_rng_methods

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, target_framework: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier.
        target_framework: Target framework identifier string.

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    if abstract_id in self.data:
      variants = self.data[abstract_id].get("variants")
      if isinstance(variants, dict):
        res = variants.get(target_framework)
        if isinstance(res, dict):
          return res
    return None

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Operation or node identifier string.

    Returns:
        True if the operation is verified, False otherwise.
    """
    return True

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Target framework identifier string.

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}


@register_hook("mock_alpha_rewrite")
def mock_plugin_logic(node: cst.Call, _ctx: HookContext) -> cst.Call:
  """Execute `register_hook` to support mock plugin logic in plugins tests.

  Args:
      node: CST or AST node under inspection or transformation.
      _ctx: HookContext passed to the plugin hook.

  Returns:
      Resulting `cst.Call` from mock plugin logic.
  """
  new_func = cst.Name("plugin_success")
  filtered: list[cst.Arg] = [a for a in node.args if not (a.keyword and a.keyword.value == "alpha")]
  filtered = cleanup_args(filtered)
  return node.with_changes(func=new_func, args=filtered)


@pytest.fixture(autouse=True)
def cleanup() -> None:
  """Create a configured `None` fixture for plugins tests."""


def test_plugin_trigger_execution() -> None:
  """Verify `ASTEngine.from_options` emits `plugin_success(x, y)` for plugin trigger execution."""
  register_hook("mock_alpha_rewrite")(mock_plugin_logic)
  mgr = MockSemantics()
  assert mgr.get_definition("torch.special_add") is not None
  engine = ASTEngine.from_options(semantics=mgr, source="torch", target="jax")
  code: str = "y = torch.special_add(x, y, alpha=0.5)"
  result: ConversionResult = engine.run(code)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_custom_framework_plugin_registration() -> None:
  """Verify `register_framework` handles custom framework plugin registration as expected."""

  @register_framework("plugin_test_fw")
  class PluginTestAdapter:
    """Sentinel `PluginTestAdapter` fixture class used in plugins tests."""

    display_name: str = "PluginTestFW"

  adapter = get_adapter("plugin_test_fw")
  assert adapter is not None
  assert isinstance(adapter, PluginTestAdapter)
  assert adapter.display_name == "PluginTestFW"
