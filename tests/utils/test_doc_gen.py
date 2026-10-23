"""Test suite for the Doc Gen module."""

from pydantic import JsonValue

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
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

  def _inject(self, name: str, tier: str, variants: dict[str, JsonValue], std_args: list[JsonValue]) -> None:
    """Mock implementation of  inject.

    Args:
        name: Identifier or operation name string.
        tier: The tier (str) for  inject.
        variants: The variants (dict[str, JsonValue]) for  inject.
        std_args: Standardized ODL parameter list.
    """
    self.data[name] = {"variants": variants, "std_args": std_args}
    self._key_origins[name] = tier

  def get_known_apis(self) -> dict[str, dict[str, JsonValue]]:
    """Mock implementation of get known apis.

    Returns:
        Resulting `dict[str, dict[str, JsonValue]]` from get known apis.
    """
    return self.data

  def get_definition_by_id(self, op_name: str) -> dict[str, JsonValue] | None:
    """Mock implementation of get definition by id.

    Args:
        op_name: Operation name string.

    Returns:
        Resulting `dict[str, JsonValue] | None` from get definition by id.
    """
    return self.data.get(op_name)


@pytest.fixture
def generator() -> MigrationGuideGenerator:
  """Execute generator helper logic.

  Returns:
      Configured `generator` fixture value (MigrationGuideGenerator).
  """
  semantics: MockSemantics = MockSemantics()
  return MigrationGuideGenerator(semantics)


def test_markdown_structure(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of markdown structure.

  Args:
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("torch", "jax")
  assert "# Migration Guide: Torch to Jax" in md
  assert "## Array" in md
  assert "## Neural" in md
  assert "| Torch API | Jax API | Argument Changes |" in md


def test_simple_match_row(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of simple match row.

  Args:
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("torch", "jax")
  assert "| `torch.abs` | `jax.numpy.abs` | - |" in md


def test_argument_diff_logic(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of argument diff logic.

  Args:
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("torch", "jax")
  assert "`jnp.sum`" in md
  assert "`input`&#8594;`a`" in md
  assert "`dim`&#8594;`axis`" in md


def test_missing_target(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of missing target.

  Args:
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("torch", "jax")
  assert "`torch.unique`" in md
  assert "| `torch.unique` | `—` |" in md


def test_plugin_annotation(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of plugin annotation.

  Args:
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("torch", "jax")
  assert "*(Plugin: magic_fix)*" in md


def test_tier_ordering(generator: MigrationGuideGenerator) -> None:
  """Verifies the behavior of tier ordering.

  Args:
      generator: Code or MLIR generator instance under test.
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
      generator: Code or MLIR generator instance under test.
  """
  md: str = generator.generate("tensorflow", "jax")
  assert "torch.abs" not in md
  assert "## Array" not in md
  assert "| `torch.abs`" not in md


def test_doc_gen_missing() -> None:
  """Verifies the behavior of documentation generation missing."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def get_definition_by_id(self, op_name: str) -> dict[str, JsonValue] | None:
      """Mock implementation of get definition by id.

      Args:
          op_name: Operation name string.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get definition by id.
      """
      if op_name == "missing":
        return None
      return {"std_args": ["a"]}

  m = MigrationGuideGenerator(DummySM())
  assert m._has_variants("missing", "jax") is False
  assert m._generate_op_row("foo", "jax", "torch") != ""


def test_doc_gen_missing_tuple_arg() -> None:
  """Verifies the behavior of documentation generation missing tuple argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def get_definition_by_id(self, op_name: str) -> dict[str, JsonValue] | None:
      """Mock implementation of get definition by id.

      Args:
          op_name: Operation name string.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get definition by id.
      """
      return {"std_args": [["a", "int"]]}

  m = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None


def test_doc_gen_missing_dict_arg() -> None:
  """Verifies the behavior of documentation generation missing dictionary argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM(SemanticsManager):
    """Dummy sm."""

    def get_definition_by_id(self, op_name: str) -> dict[str, JsonValue] | None:
      """Get definition by id.

      Args:
          op_name: Operation name string.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get definition by id.
      """
      return {"std_args": [{"name": "a", "type": "int"}]}

  m = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None
