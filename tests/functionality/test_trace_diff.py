"""Test suite for the Trace Diff module."""

import typing
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
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
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map."""
    return {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve_variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified."""
    return True


def test_conversion_trace_contains_diffs() -> None:
  """Verifies the behavior of conversion trace contains diffs."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "y = torch.abs(x)"
  result: ConversionResult = engine.run(code)
  assert result.success
  mutations: list[dict[str, typing.Any]] = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  op_event: typing.Optional[dict[str, typing.Any]] = next(
    (e for e in mutations if "Operation (abs)" in typing.cast(str, e["description"])), None
  )
  assert op_event is not None
  assert typing.cast(str, op_event["metadata"]["before"]).strip() == "torch.abs(x)"
  assert typing.cast(str, op_event["metadata"]["after"]).strip() == "jax.numpy.abs(x)"


def test_lifecycle_strip_trace() -> None:
  """Verifies the behavior of lifecycle strip trace."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code: str = "y = x.cpu()"
  result: ConversionResult = engine.run(code)
  mutations: list[dict[str, typing.Any]] = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  strip_event: typing.Optional[dict[str, typing.Any]] = next(
    (e for e in mutations if "Lifecycle Strip" in typing.cast(str, e["description"])), None
  )
  assert strip_event is not None
  assert typing.cast(str, strip_event["metadata"]["before"]).strip() == "x.cpu()"
  assert typing.cast(str, strip_event["metadata"]["after"]).strip() == "x"
