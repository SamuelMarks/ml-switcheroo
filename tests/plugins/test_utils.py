"""Test suite for the Utils module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node
from ml_switcheroo.semantics.manager import SemanticsManager


def test_create_dotted_name_simple() -> None:
  """Creates dotted name simple."""
  node = create_dotted_name("numpy")
  assert isinstance(node, cst.Name)
  assert node.value == "numpy"


def test_create_dotted_name_chained() -> None:
  """Creates dotted name chained."""
  node = create_dotted_name("jax.numpy.add")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "add"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "numpy"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "jax"


def test_detect_no_semantics() -> None:
  """Tests is_framework_module_node when semantics is None."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = HookContext(None, config)
  node = cst.Name("unknown_lib")
  assert not is_framework_module_node(node, ctx)


@pytest.fixture
def mock_ctx() -> HookContext:
  """Provides a mock ctx for testing.

  Returns:
      HookContext: Mock hook context.
  """
  semantics = MagicMock(spec=SemanticsManager)
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics.framework_configs = {
    "torch": {"alias": {"module": "torch", "name": "torch"}},
    "keras": {"alias": {"module": "keras", "name": "k"}},
    "new_lib": {},
  }
  return HookContext(semantics, config)


def test_detect_source_and_target(mock_ctx: HookContext) -> None:
  """Detects source and target.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node_torch = cst.Name("torch")
  assert is_framework_module_node(node_torch, mock_ctx)
  node_jax = cst.Name("jax")
  assert is_framework_module_node(node_jax, mock_ctx)


def test_detect_registered_framework(mock_ctx: HookContext) -> None:
  """Detects registered framework.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node = cst.Name("new_lib")
  assert is_framework_module_node(node, mock_ctx)


def test_detect_registered_alias(mock_ctx: HookContext) -> None:
  """Detects registered alias.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node = cst.Name("k")
  assert is_framework_module_node(node, mock_ctx)


def test_reject_variable(mock_ctx: HookContext) -> None:
  """Verify `Name` handles reject variable as expected.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node = cst.Name("x")
  assert not is_framework_module_node(node, mock_ctx)


def test_detect_complex_expression(mock_ctx: HookContext) -> None:
  """Detects complex expression.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert is_framework_module_node(node, mock_ctx)


def test_utils_missing() -> None:
  """Verify `Attribute` handles utils missing as expected."""
  import libcst as cst

  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.plugins.utils import _extract_root_name, is_framework_module_node

  class DummyAlias:
    """Test double `DummyAlias` implementing `model_dump` for utils tests."""

    def model_dump(self) -> dict[str, JsonValue]:
      """Mock implementation of model dump.

      Returns:
          dict: Mock data.
      """
      return {"name": "pd"}

  class DummyConf:
    """Sentinel `DummyConf` fixture class used in utils tests."""

    alias = DummyAlias()

  class _NoDumpAlias:
    """Dummy alias without model_dump."""

  class DummyConfNoDump:
    """Dummy conf no dump."""

    alias = _NoDumpAlias()

  dummy_sm = MagicMock(spec=SemanticsManager)
  dummy_sm._source_registry = {"torch.nn": ("torch", "neural")}
  dummy_sm.framework_configs = {
    "pandas": DummyConf(),
    "other": DummyConfNoDump(),
    "direct_dict": {"alias": {"name": "dd"}},
    "direct_dict_no_name": {"alias": {}},
    "no_alias": {},
    "target_framework": {},
    "tf": {},
  }
  cfg = RuntimeConfig.model_construct(source_framework="s", target_framework="target_framework")

  ctx = HookContext(
    dummy_sm,
    cfg,
  )
  assert is_framework_module_node(cst.Integer("1"), ctx) is False
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True
  assert is_framework_module_node(cst.Name("target_framework"), ctx) is True
  assert is_framework_module_node(cst.Name("tf"), ctx) is True
  assert is_framework_module_node(cst.Name("dd"), ctx) is True

  assert _extract_root_name(cst.Integer("1")) is None

  # Also test complex extraction
  attr_node = cst.Attribute(value=cst.Name("tf"), attr=cst.Name("math"))
  assert _extract_root_name(attr_node) == "tf"

  # And unknown root
  assert is_framework_module_node(cst.Name("unknown"), ctx) is False


def test_resolve_variant_helpers_attribute_error_safety() -> None:
  """Verify resolve_variant_api and resolve_variant_arg_name handle minimal contexts."""
  from ml_switcheroo.plugins.utils import resolve_variant_api, resolve_variant_arg_name

  minimal = MagicMock(spec=["lookup_api"])
  minimal.lookup_api.return_value = None
  assert resolve_variant_api(minimal, "Add") is None
  assert resolve_variant_arg_name(minimal, "lr", "learning_rate") == "learning_rate"
