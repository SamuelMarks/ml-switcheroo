"""Test suite for the Trace Diff module."""

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
    self._providers: dict[str, dict[SemanticTier | str, dict[str, str | None]]] = {}
    self._source_registry: dict[str, tuple[str, SemanticTier | str]] = {}
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
        Resulting `set[str]` from get all rng methods.
    """
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Target framework identifier string.

    Returns:
        Resulting `dict[str, tuple[str, str | None, str | None]]` from get import map.
    """
    return {}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant.

    Args:
        abstract_id: Canonical abstract operation identifier.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Target framework variant dictionary, or None if unsupported.
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
        _id: Operation or node identifier string.

    Returns:
        True if the operation is verified, False otherwise.
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
  op_event: dict[str, JsonValue] | None = next((e for e in mutations if "Operation (abs)" in str(e["description"])), None)
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
  strip_event: dict[str, JsonValue] | None = next(
    (e for e in mutations if "Lifecycle Strip" in str(e["description"])), None
  )
  assert strip_event is not None
  meta = strip_event["metadata"]
  assert isinstance(meta, dict)
  assert str(meta["before"]).strip() == "x.cpu()"
  assert str(meta["after"]).strip() == "x"
