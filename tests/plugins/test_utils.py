"""Test suite for the Utils module."""

import typing

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node
from ml_switcheroo.semantics.manager import SemanticsManager


def test_create_dotted_name_simple() -> None:
  """Creates dotted name simple."""
  node: typing.Any = create_dotted_name("numpy")
  assert isinstance(node, cst.Name)
  assert node.value == "numpy"


def test_create_dotted_name_chained() -> None:
  """Creates dotted name chained."""
  node: typing.Any = create_dotted_name("jax.numpy.add")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "add"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "numpy"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "jax"


def test_detect_no_semantics() -> None:
  """Tests is_framework_module_node when semantics is None."""
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx: typing.Any = HookContext(None, config)
  node: typing.Any = cst.Name("unknown_lib")
  assert not is_framework_module_node(node, ctx)


@pytest.fixture
def mock_ctx() -> typing.Any:
  """Provides a mock ctx for testing.

  Returns:
      HookContext: Mock hook context.
  """
  semantics: typing.Any = MagicMock(spec=SemanticsManager)
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics.framework_configs = {
    "torch": {"alias": {"module": "torch", "name": "torch"}},
    "keras": {"alias": {"module": "keras", "name": "k"}},
    "new_lib": {},
  }
  return HookContext(semantics, config)


def test_detect_source_and_target(mock_ctx: typing.Any) -> None:
  """Detects source and target.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node_torch: typing.Any = cst.Name("torch")
  assert is_framework_module_node(node_torch, mock_ctx)
  node_jax: typing.Any = cst.Name("jax")
  assert is_framework_module_node(node_jax, mock_ctx)


def test_detect_registered_framework(mock_ctx: typing.Any) -> None:
  """Detects registered framework.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: typing.Any = cst.Name("new_lib")
  assert is_framework_module_node(node, mock_ctx)


def test_detect_registered_alias(mock_ctx: typing.Any) -> None:
  """Detects registered alias.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: typing.Any = cst.Name("k")
  assert is_framework_module_node(node, mock_ctx)


def test_reject_variable(mock_ctx: typing.Any) -> None:
  """Verifies the behavior of reject variable.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: typing.Any = cst.Name("x")
  assert not is_framework_module_node(node, mock_ctx)


def test_detect_complex_expression(mock_ctx: typing.Any) -> None:
  """Detects complex expression.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: typing.Any = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert is_framework_module_node(node, mock_ctx)


# --- Merged from test_utils_missing.py ---


def test_utils_missing() -> None:
  """Verifies the behavior of utilities missing."""
  import libcst as cst

  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.plugins.utils import _extract_root_name, is_framework_module_node

  class DummyAlias:
    """Test helper class DummyAlias."""

    def model_dump(self) -> typing.Any:
      """Mock implementation of model dump.

      Returns:
          dict: Mock data.
      """
      return {"name": "pd"}

  class DummyConf:
    """Test helper class DummyConf."""

    alias: typing.Any = DummyAlias()

  class DummyConfNoDump:
    """Dummy conf no dump."""

    alias: typing.Any = object()

  class DummySM:
    """Test helper class DummySM."""

    _source_registry: typing.Any = {"torch.nn": {}}
    framework_configs: typing.Any = {
      "pandas": DummyConf(),
      "other": DummyConfNoDump(),
      "direct_dict": {"alias": {"name": "dd"}},
      "direct_dict_no_name": {"alias": {}},
      "no_alias": {},
      "target_fw": {},
      "tf": {},
    }

  class DummyConfigObj:
    """Test helper class DummyConfigObj."""

    source_framework: typing.Any = "s"
    target_framework: typing.Any = "target_fw"
    effective_source: typing.Any = "s"
    effective_target: typing.Any = "target_fw"

  ctx: typing.Any = HookContext(DummySM(), DummyConfigObj())
  assert is_framework_module_node(cst.Integer("1"), ctx) is False
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True
  assert is_framework_module_node(cst.Name("target_fw"), ctx) is True
  assert is_framework_module_node(cst.Name("tf"), ctx) is True
  assert is_framework_module_node(cst.Name("dd"), ctx) is True

  assert _extract_root_name(cst.Integer("1")) is None

  # Also test complex extraction
  attr_node: typing.Any = cst.Attribute(value=cst.Name("tf"), attr=cst.Name("math"))
  assert _extract_root_name(attr_node) == "tf"

  # And unknown root
  assert is_framework_module_node(cst.Name("unknown"), ctx) is False
