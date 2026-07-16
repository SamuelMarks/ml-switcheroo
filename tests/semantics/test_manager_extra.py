"""Auto-generated doc."""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_load_validation_report_not_found():
  """Auto-generated doc."""
  manager = SemanticsManager()
  manager.load_validation_report(Path("/non_existent_file.json"))


def test_load_validation_report_exception():
  """Auto-generated doc."""
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    path = Path(f.name)
  with patch("builtins.open", side_effect=Exception("mocked error")):
    manager.load_validation_report(path)
  path.unlink()


def test_load_validation_report_dict():
  """Auto-generated doc."""
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    json.dump({"DummyOpNonExistent": "Pass"}, f)
    path = Path(f.name)
  manager.load_validation_report(path)
  path.unlink()
  assert manager.is_verified("DummyOpNonExistent") == "Pass"


def test_is_verified_fallback():
  """Auto-generated doc."""
  manager = SemanticsManager()
  # If not tracked, defaults to True
  assert manager.is_verified("MissingOp") is True


def test_get_all_rng_methods():
  """Auto-generated doc."""
  manager = SemanticsManager()
  assert isinstance(manager.get_all_rng_methods(), set)


def test_get_patterns():
  """Auto-generated doc."""
  manager = SemanticsManager()
  assert isinstance(manager.get_patterns(), list)


def test_update_definition_inject_fields_and_validation_error():
  """Auto-generated doc."""
  manager = SemanticsManager()
  manager.update_definition("DummyOpNonExistent", {})
  manager.update_definition("DummyOpNonExistent", {"std_args": "invalid_type"})


def test_update_definition_extras_tier_and_corrupt_file_and_variants(monkeypatch):
  """Auto-generated doc."""
  manager = SemanticsManager()
  manager._key_origins["MyExtraOp"] = SemanticTier.EXTRAS.value

  import ml_switcheroo.semantics.manager as sm

  with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    monkeypatch.setattr(sm, "resolve_semantics_dir", lambda: tmppath)

    odl_dir = tmppath / "odl"
    odl_dir.mkdir(parents=True, exist_ok=True)
    file_path = odl_dir / "MyExtraOp.yaml"
    with open(file_path, "w") as f:
      f.write("corrupt yaml: [")

    manager.update_definition(
      "MyExtraOp",
      {
        "operation": "MyExtraOp",
        "description": "foo",
        "std_args": [],
        "variants": {"tensorflow": {"api": "tf.my_extra_op"}},
      },
    )

    import yaml

    with open(file_path, "r") as f:
      content = yaml.safe_load(f)
      assert content["operation"] == "MyExtraOp"


def test_update_definition_write_exception(monkeypatch):
  """Auto-generated doc."""
  manager = SemanticsManager()
  import ml_switcheroo.semantics.manager as sm

  with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    monkeypatch.setattr(sm, "resolve_semantics_dir", lambda: tmppath)

    original_open = open

    def mock_open_write(*args, **kwargs):
      """Auto-generated doc."""
      if "w" in args[1]:
        raise PermissionError("Cannot write")
      return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_write):
      manager.update_definition(
        "DummyOpNonExistent", {"operation": "DummyOpNonExistent", "description": "bar", "std_args": [], "variants": {}}
      )


def test_get_definition():
  """Auto-generated doc."""
  manager = SemanticsManager()
  manager.data["MyOp"] = {"operation": "MyOp"}
  manager._reverse_index["tf.myop"] = ("MyOp", manager.data["MyOp"])

  assert manager.get_definition("tf.myop") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("MyOp") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("NonExistent") is None
