"""Test suite for the Utils module."""

from typing import Union
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node
from ml_switcheroo.semantics.manager import SemanticsManager


def test_create_dotted_name_simple() -> None:
  """Creates dotted name simple."""
  node: Union[cst.Name, cst.Attribute] = create_dotted_name("numpy")
  assert isinstance(node, cst.Name)
  assert node.value == "numpy"


def test_create_dotted_name_chained() -> None:
  """Creates dotted name chained."""
  node: Union[cst.Name, cst.Attribute] = create_dotted_name("jax.numpy.add")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "add"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "numpy"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "jax"


@pytest.fixture
def mock_ctx() -> HookContext:
  """Provides a mock ctx for testing.

  Returns:
      HookContext: Mock hook context.
  """
  semantics: MagicMock = MagicMock(spec=SemanticsManager)
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
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
  node_torch: cst.Name = cst.Name("torch")
  assert is_framework_module_node(node_torch, mock_ctx)
  node_jax: cst.Name = cst.Name("jax")
  assert is_framework_module_node(node_jax, mock_ctx)


def test_detect_registered_framework(mock_ctx: HookContext) -> None:
  """Detects registered framework.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: cst.Name = cst.Name("new_lib")
  assert is_framework_module_node(node, mock_ctx)


def test_detect_registered_alias(mock_ctx: HookContext) -> None:
  """Detects registered alias.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: cst.Name = cst.Name("k")
  assert is_framework_module_node(node, mock_ctx)


def test_reject_variable(mock_ctx: HookContext) -> None:
  """Verifies the behavior of reject variable.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: cst.Name = cst.Name("x")
  assert not is_framework_module_node(node, mock_ctx)


def test_detect_complex_expression(mock_ctx: HookContext) -> None:
  """Detects complex expression.

  Args:
      mock_ctx (HookContext): Hook context mock.
  """
  node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert is_framework_module_node(node, mock_ctx)


# --- Merged from test_utils_missing.py ---


def test_utils_missing() -> None:
  """Verifies the behavior of utilities missing."""
  import libcst as cst

  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.plugins.utils import _extract_root_name, is_framework_module_node

  class DummyAlias:
    """Docstring."""

    def model_dump(self) -> dict:
      """Mock implementation of model dump.

      Returns:
          dict: Mock data.
      """
      return {"name": "pd"}

  class DummyConf:
    """Docstring."""

    alias: DummyAlias = DummyAlias()

  class DummyConfNoDump:
    """Dummy conf no dump."""

    alias: object = object()

  class DummySM:
    """Docstring."""

    _source_registry: dict = {"torch.nn": {}}
    framework_configs: dict = {
      "pandas": DummyConf(),
      "other": DummyConfNoDump(),
      "direct_dict": {"alias": {"name": "dd"}},
      "direct_dict_no_name": {"alias": {}},
      "no_alias": {},
      "target_fw": {},
      "tf": {},
    }

  class DummyConfigObj:
    """Docstring."""

    source_framework: str = "s"
    target_framework: str = "target_fw"
    effective_source: str = "s"
    effective_target: str = "target_fw"

  ctx: HookContext = HookContext(DummySM(), DummyConfigObj())  # type: ignore
  assert is_framework_module_node(cst.Integer("1"), ctx) is False
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True
  assert is_framework_module_node(cst.Name("target_fw"), ctx) is True
  assert is_framework_module_node(cst.Name("tf"), ctx) is True
  assert is_framework_module_node(cst.Name("dd"), ctx) is True

  assert _extract_root_name(cst.Integer("1")) is None

  # Also test complex extraction
  attr_node: cst.Attribute = cst.Attribute(value=cst.Name("tf"), attr=cst.Name("math"))
  assert _extract_root_name(attr_node) == "tf"

  # And unknown root
  assert is_framework_module_node(cst.Name("unknown"), ctx) is False
