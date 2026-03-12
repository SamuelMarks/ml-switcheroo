def test_injector_spec_write_parent_not_exist():
  from ml_switcheroo.tools.injector_spec import StandardsInjector
  from ml_switcheroo.core.dsl import OperationDef

  op_def = OperationDef(operation="Foo", description="Foo", variants={})
  injector = StandardsInjector(op_def)

  # 101: if not target_path.parent.exists()
  import pathlib

  # Let's use a real temporary directory
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "nested" / "file.json"

    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.tools.injector_spec.resolve_semantics_dir", return_value=p
    ):
      injector.inject(dry_run=False)
      assert p.exists()
