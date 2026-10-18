"""Test suite for the Doc Gen module."""

import typing

from typing import Any, Dict, List, Optional

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: Dict[str, Any] = {}
    self._reverse_index: Dict[str, Any] = {}
    self._key_origins: Dict[str, str] = {}
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

  def _inject(self, name: str, tier: str, variants: Dict[str, Any], std_args: List[Any]) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        tier: Test argument or fixture.
        variants: Test argument or fixture.
        std_args: Test argument or fixture.
    """
    self.data[name] = {"variants": variants, "std_args": std_args}
    self._key_origins[name] = tier

  def get_known_apis(self) -> Dict[str, Any]:
    """Mock implementation of get known apis.

    Returns:
        Computed test result or fixture value.
    """
    return self.data

  def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
    """Mock implementation of get definition by id.

    Args:
        op_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(op_name)


@pytest.fixture
def generator() -> MigrationGuideGenerator:
  """Execute generator helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics: MockSemantics = MockSemantics()
  return MigrationGuideGenerator(semantics)


def test_markdown_structure(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of markdown structure.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  assert "# Migration Guide: Torch to Jax" in md
  assert "## Array" in md
  assert "## Neural" in md
  assert "| Torch API | Jax API | Argument Changes |" in md


def test_simple_match_row(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of simple match row.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  assert "| `torch.abs` | `jax.numpy.abs` | - |" in md


def test_argument_diff_logic(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of argument diff logic.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  assert "`jnp.sum`" in md
  assert "`input`&#8594;`a`" in md
  assert "`dim`&#8594;`axis`" in md


def test_missing_target(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of missing target.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  assert "`torch.unique`" in md
  assert "| `torch.unique` | `—` |" in md


def test_plugin_annotation(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of plugin annotation.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  assert "*(Plugin: magic_fix)*" in md


def test_tier_ordering(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of tier ordering.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("torch", "jax")
  idx_array: int = md.find("## Array")
  idx_neural: int = md.find("## Neural")
  idx_extras: int = md.find("## Extras")
  assert idx_array < idx_neural
  assert idx_neural < idx_extras


def test_filtering_missing_source(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of filtering missing source.

  Args:
      generator: Test argument or fixture.
  """
  md: str = generator.generate("tensorflow", "jax")
  assert "torch.abs" not in md
  assert "## Array" not in md
  assert "| `torch.abs`" not in md


# --- Merged from test_doc_gen_missing.py ---


def test_doc_gen_missing() -> None:
  """Verifies the behavior of documentation generation missing."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Test helper class DummySM."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Mock implementation of get definition by id.

      Args:
          op_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if op_name == "missing":
        return None
      return {"std_args": ["a"]}

  m: typing.Any = MigrationGuideGenerator(DummySM())
  assert m._has_variants("missing", "jax") is False
  assert m._generate_op_row("foo", "jax", "torch") != ""


def test_doc_gen_missing_tuple_arg() -> None:
  """Verifies the behavior of documentation generation missing tuple argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Test helper class DummySM."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Mock implementation of get definition by id.

      Args:
          op_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"std_args": [("a", "int")]}

  m: typing.Any = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None


def test_doc_gen_missing_dict_arg() -> None:
  """Verifies the behavior of documentation generation missing dictionary argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy sm."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Get definition by id.

      Args:
          op_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"std_args": [{"name": "a", "type": "int"}]}

  m: typing.Any = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None
