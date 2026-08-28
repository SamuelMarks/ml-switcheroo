"""Test suite for the Doc Gen Missing module."""

from typing import Dict, Any, Optional


def test_doc_gen_missing() -> None:
  """Verifies the behavior of documentation generation missing."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy S M class for testing purposes."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Mock implementation of get definition by id."""
      if op_name == "missing":
        return None
      return {"std_args": ["a"]}

  m: MigrationGuideGenerator = MigrationGuideGenerator(DummySM())
  assert m._has_variants("missing", "jax") is False
  assert m._generate_op_row("foo", "jax", "torch") != ""


def test_doc_gen_missing_tuple_arg() -> None:
  """Verifies the behavior of documentation generation missing tuple argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy S M class for testing purposes."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Mock implementation of get definition by id."""
      return {"std_args": [("a", "int")]}

  m: MigrationGuideGenerator = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None


def test_doc_gen_missing_dict_arg() -> None:
  """Verifies the behavior of documentation generation missing dictionary argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy sm."""

    def get_definition_by_id(self, op_name: str) -> Optional[Dict[str, Any]]:
      """Get definition by id."""
      return {"std_args": [{"name": "a", "type": "int"}]}

  m: MigrationGuideGenerator = MigrationGuideGenerator(DummySM())
  res: str = m._generate_op_row("foo", "jax", "torch")
  assert res is not None
