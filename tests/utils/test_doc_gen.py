"""Test suite for the Doc Gen module."""

import pytest
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._inject(
      "abs", tier="array", variants={"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}, std_args=["x"]
    )
    self._inject(
      "sum",
      tier="neural",
      variants={
        "torch": {"api": "torch.sum", "args": {"x": "input", "axis": "dim"}},
        "jax": {"api": "jnp.sum", "args": {"x": "a", "axis": "axis"}},
      },
      std_args=["x", "axis"],
    )
    self._inject("unique_op", tier="extras", variants={"torch": {"api": "torch.unique"}}, std_args=["x"])
    self._inject(
      "complex_op",
      tier="array",
      variants={"torch": {"api": "torch.complex"}, "jax": {"api": "jax.complex", "requires_plugin": "magic_fix"}},
      std_args=["x"],
    )

  def _inject(self, name, tier, variants, std_args):
    """Mock implementation of  inject."""
    self.data[name] = {"variants": variants, "std_args": std_args}
    self._key_origins[name] = tier

  def get_known_apis(self):
    """Mock implementation of get known apis."""
    return self.data

  def get_definition_by_id(self, op_name):
    """Mock implementation of get definition by id."""
    return self.data.get(op_name)


@pytest.fixture
def generator():
  """Provides a mock generator for testing."""
  semantics = MockSemantics()
  return MigrationGuideGenerator(semantics)


def test_markdown_structure(generator):
  """Verifies the behavior of markdown structure."""
  md = generator.generate("torch", "jax")
  assert "# Migration Guide: Torch to Jax" in md
  assert "## Array" in md
  assert "## Neural" in md
  assert "| Torch API | Jax API | Argument Changes |" in md


def test_simple_match_row(generator):
  """Verifies the behavior of simple match row."""
  md = generator.generate("torch", "jax")
  assert "| `torch.abs` | `jax.numpy.abs` | - |" in md


def test_argument_diff_logic(generator):
  """Verifies the behavior of argument diff logic."""
  md = generator.generate("torch", "jax")
  assert "`jnp.sum`" in md
  assert "`input`&#8594;`a`" in md
  assert "`dim`&#8594;`axis`" in md


def test_missing_target(generator):
  """Verifies the behavior of missing target."""
  md = generator.generate("torch", "jax")
  assert "`torch.unique`" in md
  assert "| `torch.unique` | `—` |" in md


def test_plugin_annotation(generator):
  """Verifies the behavior of plugin annotation."""
  md = generator.generate("torch", "jax")
  assert "*(Plugin: magic_fix)*" in md


def test_tier_ordering(generator):
  """Verifies the behavior of tier ordering."""
  md = generator.generate("torch", "jax")
  idx_array = md.find("## Array")
  idx_neural = md.find("## Neural")
  idx_extras = md.find("## Extras")
  assert idx_array < idx_neural
  assert idx_neural < idx_extras


def test_filtering_missing_source(generator):
  """Verifies the behavior of filtering missing source."""
  md = generator.generate("tensorflow", "jax")
  assert "torch.abs" not in md
  assert "## Array" not in md
  assert "| `torch.abs`" not in md
