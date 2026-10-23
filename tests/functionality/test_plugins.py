"""Test suite for the Plugins module."""

import typing

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.frameworks.base import get_adapter, register_framework
from ml_switcheroo.semantics.manager import SemanticsManager


def cleanup_args(args_list: list[cst.Arg]) -> list[cst.Arg]:
  """Helper to cleanup arguments.

  Args:
      args_list: The args list (list[cst.Arg]) for cleanup args.

  Returns:
      Resulting `list[cst.Arg]` from cleanup args.
  """
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

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
    special_def = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    getattr(self.data, "__setitem__")("special_add", special_def)
    getattr(self._reverse_index, "__setitem__")(
      "torch.special_add",
      ("special_add", special_def),
    )
    add_def = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    getattr(self.data, "__setitem__")("add", add_def)
    getattr(self._reverse_index, "__setitem__")(
      "jax.numpy.add",
      ("add", add_def),
    )
    getattr(self._reverse_index, "__setitem__")(
      "torch.add",
      ("add", add_def),
    )

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
  """Execute mock plugin logic helper logic.

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
def cleanup() -> typing.Generator[None, None, None]:
  """Helper to cleanup.

  Returns:
      Configured `cleanup` fixture value (typing.Generator[None, None, None]).
  """
  yield
  pass


def test_plugin_trigger_execution() -> None:
  """Verifies the behavior of plugin trigger execution."""
  register_hook("mock_alpha_rewrite")(mock_plugin_logic)
  mgr = MockSemantics()
  assert mgr.get_definition("torch.special_add") is not None
  engine = ASTEngine(semantics=mgr, source="torch", target="jax")
  code: str = "y = torch.special_add(x, y, alpha=0.5)"
  result: ConversionResult = engine.run(code)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_custom_framework_plugin_registration() -> None:
  """Verifies the behavior of custom framework plugin registration."""

  @register_framework("plugin_test_fw")
  class PluginTestAdapter:
    """Test helper class PluginTestAdapter."""

    pass

  adapter = get_adapter("plugin_test_fw")
  assert adapter is not None
