"""Test suite for the Engine Linter module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager


class MockGlobalUsageScanner(cst.CSTVisitor):
  """Test helper class MockGlobalUsageScanner."""

  def __init__(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Initializes the MockGlobalUsageScanner instance.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.
    """
    self.used_names = {"torch"}

  def on_visit(self, node: cst.CSTNode) -> bool:
    """Mock implementation of on visit.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return False

  def on_leave(self, node: cst.CSTNode) -> None:
    """Mock implementation of on leave.

    Args:
        node: Test argument or fixture.
    """
    pass


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.framework_configs = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self.test_templates = {}
    self._known_rng_methods: set[str] = set()
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self._validation_status = {}
    self._providers = {}
    self._source_registry = {}

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_framework_aliases(self) -> dict[str, tuple[str, str]]:
    """Mock implementation of get framework aliases.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return set()

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def engine() -> typing.Generator[ASTEngine, None, None]:
  """Execute engine helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  mock_torch = MagicMock()
  mock_torch.configure_mock(import_alias=("torch", "torch"), inherits_from=None)
  del mock_torch.create_emitter
  del mock_torch.create_parser
  mock_jax = MagicMock()
  mock_jax.configure_mock(import_alias=("jax.numpy", "jnp"), inherits_from=None)
  del mock_jax.create_emitter
  del mock_jax.create_parser

  def get_adapter_side_effect(name: str) -> MagicMock | None:
    """Gets adapter side effect.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "torch":
      return mock_torch
    if name == "jax":
      return mock_jax
    return None

  with patch("ml_switcheroo.frameworks.get_adapter", side_effect=get_adapter_side_effect):
    yield ASTEngine(semantics=mgr, config=config)


def test_engine_catches_leaked_import(engine: ASTEngine) -> None:
  """Verifies the behavior of engine catches leaked import."""
  code: str = "\nimport torch\nx = torch.add(1, 2)\n"
  with patch("ml_switcheroo.core.engine.GlobalUsageScanner", side_effect=MockGlobalUsageScanner):
    result: ConversionResult = engine.run(code)
  assert result.success is True
  assert result.errors is not None
  assert len(result.errors) > 0
  assert any(("Forbidden Import: 'torch'" in e for e in result.errors))


def test_engine_catches_leaked_usage(engine: ASTEngine) -> None:
  """Verifies the behavior of engine catches leaked usage."""
  code: str = "\nimport torch\ny = torch.abs(x)\n"
  result: ConversionResult = engine.run(code)
  assert "torch.abs(x)" in result.code
  assert result.has_errors
  errors_str: str = str(result.errors)
  assert "Forbidden" in errors_str


def test_linter_trace_event(engine: ASTEngine) -> None:
  """Verifies the behavior of linter trace event.

  Args:
      engine: Test argument or fixture.
  """
  code: str = "import torch"
  with patch("ml_switcheroo.core.engine.GlobalUsageScanner", side_effect=MockGlobalUsageScanner):
    result: ConversionResult = engine.run(code)
  phase_descriptions: list[str] = [str(e["description"]) for e in result.trace_events if e["type"] == "phase_start"]
  assert "Structural Linter" in phase_descriptions
