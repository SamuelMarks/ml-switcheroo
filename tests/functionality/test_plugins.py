"""Test suite for the Plugins module."""

import libcst as cst
import pytest
from typing import Set, Dict, Tuple, Optional
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import register_hook, _HOOKS
from ml_switcheroo.frameworks.base import register_framework, get_adapter


def cleanup_args(args_list):
  """Helper to cleanup arguments."""
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.framework_configs = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self._providers = {}
    self._source_registry = {}
    special_def = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    self.data["special_add"] = special_def
    self._reverse_index["torch.special_add"] = ("special_add", special_def)
    add_def = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self.data["add"] = add_def
    self._reverse_index["jax.numpy.add"] = ("add", add_def)
    self._reverse_index["torch.add"] = ("add", add_def)

  def get_all_rng_methods(self) -> Set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id, target_fw):
    """Mock implementation of resolve variant."""
    if abstract_id in self.data:
      return self.data[abstract_id]["variants"].get(target_fw)
    return None

  def is_verified(self, _id):
    """Mock implementation of is verified."""
    return True

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map."""
    return {}


@register_hook("mock_alpha_rewrite")
def mock_plugin_logic(node, _ctx):
  """Provides a mock plugin logic for testing."""
  new_func = cst.Name("plugin_success")
  filtered = [a for a in node.args if not (a.keyword and a.keyword.value == "alpha")]
  filtered = cleanup_args(filtered)
  return node.with_changes(func=new_func, args=filtered)


@pytest.fixture(autouse=True)
def cleanup():
  """Helper to cleanup."""
  yield
  pass


def test_plugin_trigger_execution():
  """Verifies the behavior of plugin trigger execution."""
  _HOOKS["mock_alpha_rewrite"] = mock_plugin_logic
  mgr = MockSemantics()
  assert mgr.get_definition("torch.special_add") is not None
  engine = ASTEngine(semantics=mgr, source="torch", target="jax")
  code = "y = torch.special_add(x, y, alpha=0.5)"
  result = engine.run(code)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_custom_framework_plugin_registration():
  """Verifies the behavior of custom framework plugin registration."""

  @register_framework("plugin_test_fw")
  class PluginTestAdapter:
    """Test suite for the Plugin Test Adapter component."""

    pass

  adapter = get_adapter("plugin_test_fw")
  assert adapter is not None
