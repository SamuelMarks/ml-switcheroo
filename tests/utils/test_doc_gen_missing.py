"""Test suite for the Doc Gen Missing module."""


def test_doc_gen_missing():
  """Verifies the behavior of documentation generation missing."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy S M class for testing purposes."""

    def get_definition_by_id(self, op_name):
      """Mock implementation of get definition by id."""
      if op_name == "missing":
        return None
      return {"std_args": ["a"]}

  m = MigrationGuideGenerator(DummySM())
  assert m._has_variants("missing", "jax") is False
  assert m._generate_op_row("foo", "jax", "torch") != ""


def test_doc_gen_missing_tuple_arg():
  """Verifies the behavior of documentation generation missing tuple argument."""
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    """Dummy S M class for testing purposes."""

    def get_definition_by_id(self, op_name):
      """Mock implementation of get definition by id."""
      return {"std_args": [("a", "int")]}

  m = MigrationGuideGenerator(DummySM())
  res = m._generate_op_row("foo", "jax", "torch")
  assert res is not None
