"""Test suite for the Injector Spec Missing2 module."""

import pathlib


def test_injector_spec_write_parent_not_exist() -> None:
  """Verifies the behavior of injector spec write parent not exist."""
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef

  op_def: OperationDef = OperationDef(operation="Foo", description="Foo", variants={})
  injector: StandardsInjector = StandardsInjector(op_def)
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p: pathlib.Path = pathlib.Path(td) / "nested" / "file.json"
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=p
    ):
      injector.inject(dry_run=False)
      assert p.exists()
