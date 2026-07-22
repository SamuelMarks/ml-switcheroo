"""Test suite for the Utils module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


def test_create_dotted_name_simple():
  """Creates dotted name simple."""
  node = create_dotted_name("numpy")
  assert isinstance(node, cst.Name)
  assert node.value == "numpy"


def test_create_dotted_name_chained():
  """Creates dotted name chained."""
  node = create_dotted_name("jax.numpy.add")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "add"
  assert node.value.attr.value == "numpy"
  assert node.value.value.value == "jax"


@pytest.fixture
def mock_ctx():
  """Provides a mock ctx for testing."""
  semantics = MagicMock(spec=SemanticsManager)
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics.framework_configs = {
    "torch": {"alias": {"module": "torch", "name": "torch"}},
    "keras": {"alias": {"module": "keras", "name": "k"}},
    "new_lib": {},
  }
  return HookContext(semantics, config)


def test_detect_source_and_target(mock_ctx):
  """Detects source and target."""
  node_torch = cst.Name("torch")
  assert is_framework_module_node(node_torch, mock_ctx)
  node_jax = cst.Name("jax")
  assert is_framework_module_node(node_jax, mock_ctx)


def test_detect_registered_framework(mock_ctx):
  """Detects registered framework."""
  node = cst.Name("new_lib")
  assert is_framework_module_node(node, mock_ctx)


def test_detect_registered_alias(mock_ctx):
  """Detects registered alias."""
  node = cst.Name("k")
  assert is_framework_module_node(node, mock_ctx)


def test_reject_variable(mock_ctx):
  """Verifies the behavior of reject variable."""
  node = cst.Name("x")
  assert not is_framework_module_node(node, mock_ctx)


def test_detect_complex_expression(mock_ctx):
  """Detects complex expression."""
  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert is_framework_module_node(node, mock_ctx)
