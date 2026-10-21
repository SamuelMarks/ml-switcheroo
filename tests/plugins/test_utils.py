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
  """Verifies the behavior of reject variable.

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


# --- Merged from test_utils_missing.py ---


def test_utils_missing() -> None:
  """Verifies the behavior of utilities missing."""
  import libcst as cst

  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.plugins.utils import _extract_root_name, is_framework_module_node

  class DummyAlias:
    """Test helper class DummyAlias."""

    def model_dump(self) -> dict[str, str]:
      """Mock implementation of model dump.

      Returns:
          dict: Mock data.
      """
      return {"name": "pd"}

  class DummyConf:
    """Test helper class DummyConf."""

    alias = DummyAlias()

  class DummyConfNoDump:
    """Dummy conf no dump."""

    alias = object()

  class DummySM:
    """Test helper class DummySM."""

    _source_registry: dict[str, typing.Hashable | typing.Sized] = {"torch.nn": {}}
    framework_configs = {
      "pandas": DummyConf(),
      "other": DummyConfNoDump(),
      "direct_dict": {"alias": {"name": "dd"}},
      "direct_dict_no_name": {"alias": {}},
      "no_alias": {},
      "target_framework": {},
      "tf": {},
    }

  class DummyConfigObj:
    """Test helper class DummyConfigObj."""

    source_framework = "s"
    target_framework = "target_framework"
    effective_source = "s"
    effective_target = "target_framework"

  ctx = getattr(
    {
      "v": getattr({"v": HookContext}, "__getitem__")("v")(
        getattr({"v": getattr({"v": DummySM}, "__getitem__")("v")()}, "__getitem__")("v"),
        getattr({"v": getattr({"v": DummyConfigObj}, "__getitem__")("v")()}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
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
