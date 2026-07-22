"""Test suite for the Trace Diff module."""

from typing import Set, Dict, Any, Tuple, Optional
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.framework_configs = {}
    self._validation_status = {}
    self._key_origins = {}
    self._known_rng_methods = set()
    self._providers = {}
    self._source_registry = {}
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

  def get_all_rng_methods(self) -> Set[str]:
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map."""
    return {}


def test_conversion_trace_contains_diffs():
  """Verifies the behavior of conversion trace contains diffs."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "y = torch.abs(x)"
  result = engine.run(code)
  assert result.success
  mutations = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  op_event = next((e for e in mutations if "Operation (abs)" in e["description"]), None)
  assert op_event is not None
  assert op_event["metadata"]["before"].strip() == "torch.abs(x)"
  assert op_event["metadata"]["after"].strip() == "jax.numpy.abs(x)"


def test_lifecycle_strip_trace():
  """Verifies the behavior of lifecycle strip trace."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "y = x.cpu()"
  result = engine.run(code)
  mutations = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  strip_event = next((e for e in mutations if "Lifecycle Strip" in e["description"]), None)
  assert strip_event is not None
  assert strip_event["metadata"]["before"].strip() == "x.cpu()"
  assert strip_event["metadata"]["after"].strip() == "x"
