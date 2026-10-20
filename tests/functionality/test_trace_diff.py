"""Test suite for the Trace Diff module."""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, dict[str, JsonValue]] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.framework_configs: dict[str, dict[str, JsonValue]] = {}
    self._validation_status: dict[str, bool] = {}
    self._key_origins: dict[str, str] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, dict[typing.Union[SemanticTier, str], dict[str, typing.Optional[str]]]] = {}
    self._source_registry: dict[str, tuple[str, typing.Union[SemanticTier, str]]] = {}
    self.data["abs"] = {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}, "std_args": ["x"]}
    self._reverse_index["torch.abs"] = ("abs", self.data["abs"])
    self.framework_configs = {
      "torch": {
        "traits": {
          "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
          "lifecycle_warn_methods": ["eval", "train"],
        }
      }
    }

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Mock resolve_variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    variants = self.data.get(abstract_id, {}).get("variants")
    if isinstance(variants, dict):
      res = variants.get(fw)
      if isinstance(res, dict):
        return res
    return None

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True


def test_conversion_trace_contains_diffs() -> None:
  """Verifies the behavior of conversion trace contains diffs."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "y = torch.abs(x)"
  result: ConversionResult = engine.run(code)
  assert result.success
  mutations: list[dict[str, JsonValue]] = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  op_event: typing.Optional[dict[str, JsonValue]] = next(
    (e for e in mutations if "Operation (abs)" in str(e["description"])), None
  )
  assert op_event is not None
  meta = op_event["metadata"]
  assert isinstance(meta, dict)
  assert str(meta["before"]).strip() == "torch.abs(x)"
  assert str(meta["after"]).strip() == "jax.numpy.abs(x)"


def test_lifecycle_strip_trace() -> None:
  """Verifies the behavior of lifecycle strip trace."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "y = x.cpu()"
  result: ConversionResult = engine.run(code)
  mutations: list[dict[str, JsonValue]] = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  strip_event: typing.Optional[dict[str, JsonValue]] = next(
    (e for e in mutations if "Lifecycle Strip" in str(e["description"])), None
  )
  assert strip_event is not None
  meta = strip_event["metadata"]
  assert isinstance(meta, dict)
  assert str(meta["before"]).strip() == "x.cpu()"
  assert str(meta["after"]).strip() == "x"
