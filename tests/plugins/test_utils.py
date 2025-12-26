"""
Tests for Plugin Utilities (Dynamic Framework Detection).

Verifies:
1. create_dotted_name builds correct CST.
2. is_framework_module_node detects configured source/target frameworks.
3. is_framework_module_node detection dynamically checks Registered Semantics.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


def test_create_dotted_name_simple():
  node = create_dotted_name("numpy")
  assert isinstance(node, cst.Name)
  assert node.value == "numpy"


def test_create_dotted_name_chained():
  node = create_dotted_name("jax.numpy.add")
  # Structure: Attribute(value=Attribute(value=Name(jax), attr=numpy), attr=add)
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "add"
  assert node.value.attr.value == "numpy"
  assert node.value.value.value == "jax"


@pytest.fixture
def mock_ctx():
  semantics = MagicMock(spec=SemanticsManager)
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  # Configure mock semantics with some registered frameworks
  semantics.framework_configs = {
    "torch": {"alias": {"module": "torch", "name": "torch"}},
    "keras": {"alias": {"module": "keras", "name": "k"}},
    "new_lib": {},  # No alias
  }

  return HookContext(semantics, config)


def test_detect_source_and_target(mock_ctx):
  """Verify source_fw and target_fw from config are detected."""
  # torch (source)
  node_torch = cst.Name("torch")
  assert is_framework_module_node(node_torch, mock_ctx)

  # jax (target)
  node_jax = cst.Name("jax")
  assert is_framework_module_node(node_jax, mock_ctx)


def test_detect_registered_framework(mock_ctx):
  """Verify frameworks in registry are detected."""
  node = cst.Name("new_lib")
  assert is_framework_module_node(node, mock_ctx)


def test_detect_registered_alias(mock_ctx):
  """Verify aliases in registry configuration are detected."""
  # keras has alias 'k' in fixture
  node = cst.Name("k")
  assert is_framework_module_node(node, mock_ctx)


def test_reject_variable(mock_ctx):
  """Verify random variables are rejected."""
  node = cst.Name("x")
  assert not is_framework_module_node(node, mock_ctx)


def test_detect_complex_expression(mock_ctx):
  """Verify dot-separated modules are detected by root."""
  # Check `torch.nn`
  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert is_framework_module_node(node, mock_ctx)
