"""Test suite for the Plugins module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import _HOOKS, register_hook
from ml_switcheroo.frameworks.base import get_adapter, register_framework
from ml_switcheroo.semantics.manager import SemanticsManager


def cleanup_args(args_list: list[cst.Arg]) -> list[cst.Arg]:
  """Helper to cleanup arguments."""
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    special_def: dict[str, typing.Any] = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    self.data["special_add"] = special_def
    self._reverse_index["torch.special_add"] = ("special_add", special_def)
    add_def: dict[str, typing.Any] = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self.data["add"] = add_def
    self._reverse_index["jax.numpy.add"] = ("add", add_def)
    self._reverse_index["torch.add"] = ("add", add_def)

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, target_fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Mock implementation of resolve variant."""
    if abstract_id in self.data:
      return self.data[abstract_id]["variants"].get(target_fw)
    return None

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified."""
    return True

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map."""
    return {}


@register_hook("mock_alpha_rewrite")
def mock_plugin_logic(node: cst.Call, _ctx: typing.Any) -> cst.Call:
  """Docstring."""
  new_func = cst.Name("plugin_success")
  filtered: list[cst.Arg] = [
    a for a in node.args if not (a.keyword and typing.cast(cst.Name, a.keyword).value == "alpha")
  ]
  filtered = cleanup_args(filtered)
  return node.with_changes(func=new_func, args=filtered)


@pytest.fixture(autouse=True)
def cleanup() -> typing.Generator[None, None, None]:
  """Helper to cleanup."""
  yield
  pass


def test_plugin_trigger_execution() -> None:
  """Verifies the behavior of plugin trigger execution."""
  _HOOKS["mock_alpha_rewrite"] = mock_plugin_logic
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
    pass

  adapter: typing.Any = get_adapter("plugin_test_fw")
  assert adapter is not None
