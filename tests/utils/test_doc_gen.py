"""
Tests for Migration Guide Generator.

Verifies that:
1. Markdown is generated with correct headers.
2. Argument mapping diffs are calculated correctly (dim -> axis).
3. Missing API variants are handled gracefully.
4. Plugins are noted in the table.
5. Operations are grouped by Semantic Tier.
"""

import pytest
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager with deterministic data for documentation generation.
  """

  def __init__(self):
    # Initialize empty
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}

    # 1. Simple Match (Array Tier)
    self._inject(
      "abs",
      tier="array",
      variants={
        "torch": {"api": "torch.abs"},
        "jax": {"api": "jax.numpy.abs"},
      },
      std_args=["x"],
    )

    # 2. Argument Rename (Neural Tier)
    # sum(x, axis) -> torch: sum(input, dim), jax: sum(a, axis)
    self._inject(
      "sum",
      tier="neural",
      variants={
        "torch": {"api": "torch.sum", "args": {"x": "input", "axis": "dim"}},
        "jax": {"api": "jnp.sum", "args": {"x": "a", "axis": "axis"}},
      },
      std_args=["x", "axis"],
    )

    # 3. Valid Source, Missing Target
    self._inject(
      "unique_op",
      tier="extras",
      variants={"torch": {"api": "torch.unique"}},
      std_args=["x"],
    )

    # 4. Plugin Usage
    self._inject(
      "complex_op",
      tier="array",
      variants={
        "torch": {"api": "torch.complex"},
        "jax": {"api": "jax.complex", "requires_plugin": "magic_fix"},
      },
      std_args=["x"],
    )

  def _inject(self, name, tier, variants, std_args):
    self.data[name] = {"variants": variants, "std_args": std_args}
    self._key_origins[name] = tier

  def get_known_apis(self):
    return self.data

  def get_definition_by_id(self, op_name):
    return self.data.get(op_name)


@pytest.fixture
def generator():
  semantics = MockSemantics()
  return MigrationGuideGenerator(semantics)


def test_markdown_structure(generator):
  """
  Verify overall document structure (Headers, Intro).
  """
  md = generator.generate("torch", "jax")

  assert "# Migration Guide: Torch to Jax" in md
  assert "## Array" in md
  assert "## Neural" in md
  assert "| Torch API | Jax API | Argument Changes |" in md


def test_simple_match_row(generator):
  """
  Verify a simple 1:1 mapping produces a clean row.
  """
  md = generator.generate("torch", "jax")
  # | `torch.abs` | `jax.numpy.abs` | - |
  assert "| `torch.abs` | `jax.numpy.abs` | - |" in md


def test_argument_diff_logic(generator):
  """
  Verify argument renaming diffs are computed.
  Std: x, axis
  Torch: input, dim
  JAX: a, axis

  Diffs:
  - input -> a
  - dim -> axis (Target uses default 'axis' which matches std 'axis', so Diff is 'dim' -> 'axis')
  """
  md = generator.generate("torch", "jax")

  # Check for the sum row
  assert "`jnp.sum`" in md
  # Check diff strings
  assert "`input`&#8594;`a`" in md
  assert "`dim`&#8594;`axis`" in md


def test_missing_target(generator):
  """
  Verify operations missing in target show up with '—'.
  """
  md = generator.generate("torch", "jax")

  # unique_op: Torch API exists, Jax API missing
  assert "`torch.unique`" in md
  assert "| `torch.unique` | `—` |" in md


def test_plugin_annotation(generator):
  """
  Verify operations using plugins are annotated.
  """
  md = generator.generate("torch", "jax")

  assert "*(Plugin: magic_fix)*" in md


def test_tier_ordering(generator):
  """
  Verify Tiers appear in specific order: Array, Neural, Extras.
  """
  md = generator.generate("torch", "jax")

  idx_array = md.find("## Array")
  idx_neural = md.find("## Neural")
  idx_extras = md.find("## Extras")

  assert idx_array < idx_neural
  assert idx_neural < idx_extras


def test_filtering_missing_source(generator):
  """
  Verify that if Source FW does not have the op, it is skipped entirely.
  If source is 'tensorflow' (unknown in mock), doc should represent that nothing was found
  (headers skipped because no ops valid).
  """
  # Ask for conversion FROM 'tensorflow' (which has no definitions in our mock)
  md = generator.generate("tensorflow", "jax")

  # Should not list torch ops
  assert "torch.abs" not in md
  # Should contain NO tables because no ops match source 'tensorflow'
  assert "## Array" not in md
  assert "| `torch.abs`" not in md
