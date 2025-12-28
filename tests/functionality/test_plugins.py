"""
Tests for Plugin Logic and Hook Execution Infrastructure.

Verifies:
1. Hook registration and retrieval.
2. Hook execution via ASTEngine.
3. Context object integration.
"""

import libcst as cst
import pytest
from unittest.mock import MagicMock
from typing import Set, Dict, Tuple, Optional

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import register_hook, _HOOKS, clear_hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework, get_adapter


# Reused helper to clean lists
def cleanup_args(args_list):
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  """
  Mock Semantics Manager for testing plugin triggers.
  """

  def __init__(self):
    """Initialize with specific mock data for 'special_add'."""
    # Do NOT call super().__init__(), it loads real files and resets data
    self.data = {}
    self._reverse_index = {}
    self.framework_configs = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # New attributes for import abstraction
    self._providers = {}
    self._source_registry = {}

    # 1. 'special_add' - Wired to 'mock_alpha_rewrite' plugin
    special_def = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    self.data["special_add"] = special_def
    self._reverse_index["torch.special_add"] = ("special_add", special_def)

    # 2. 'add' - Standard mapping
    add_def = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self.data["add"] = add_def
    self._reverse_index["jax.numpy.add"] = ("add", add_def)
    self._reverse_index["torch.add"] = ("add", add_def)

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id, target_fw):
    if abstract_id in self.data:
      return self.data[abstract_id]["variants"].get(target_fw)
    return None

  def is_verified(self, _id):
    return True

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    return {}


# Local mock implementation of a plugin logic
@register_hook("mock_alpha_rewrite")
def mock_plugin_logic(node, _ctx):
  """
  Rewrite to 'plugin_success(x, y)' removing alpha arg.
  """
  new_func = cst.Name("plugin_success")
  # Filter alpha
  filtered = [a for a in node.args if not (a.keyword and a.keyword.value == "alpha")]
  filtered = cleanup_args(filtered)
  return node.with_changes(func=new_func, args=filtered)


@pytest.fixture(autouse=True)
def cleanup():
  # Ensure hooks are cleared to avoid pollution from other tests
  yield
  clear_hooks()


def test_plugin_trigger_execution():
  """
  Verify that an operation mapped to 'requires_plugin' actually triggers the hook.
  """
  # Force registration of our local mock
  _HOOKS["mock_alpha_rewrite"] = mock_plugin_logic

  mgr = MockSemantics()
  # Check integrity before run
  assert mgr.get_definition("torch.special_add") is not None

  engine = ASTEngine(semantics=mgr, source="torch", target="jax")

  # Input has 'alpha' argument
  code = "y = torch.special_add(x, y, alpha=0.5)"

  result = engine.run(code)

  # Expect clean syntax: plugin_success(x, y)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_custom_framework_plugin_registration():
  """
  Verify that we can register a custom framework adapter that might rely on plugins.
  """

  @register_framework("plugin_test_fw")
  class PluginTestAdapter:
    pass

  adapter = get_adapter("plugin_test_fw")
  assert adapter is not None
