def test_doc_gen_missing():
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
  from ml_switcheroo.semantics.manager import SemanticsManager

  class DummySM:
    def get_definition_by_id(self, op_name):
      if op_name == "missing":
        return None
      return {"std_args": ["a"]}

  m = MigrationGuideGenerator(DummySM())
  assert m._has_variants("missing", "jax") is False

  assert m._generate_op_row("foo", "jax", "torch") != ""


def test_doc_gen_missing_tuple_arg():
  from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator

  class DummySM:
    def get_definition_by_id(self, op_name):
      return {"std_args": [("a", "int")]}

  m = MigrationGuideGenerator(DummySM())
  res = m._generate_op_row("foo", "jax", "torch")
  assert res is not None
