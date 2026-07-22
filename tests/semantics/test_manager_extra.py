"""Test suite for the Manager Extra module."""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_load_validation_report_not_found():
  """Loads validation report not found."""
  manager = SemanticsManager()
  manager.load_validation_report(Path("/non_existent_file.json"))


def test_load_validation_report_exception():
  """Loads validation report correctly handling an exception."""
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    path = Path(f.name)
  with patch("builtins.open", side_effect=Exception("mocked error")):
    manager.load_validation_report(path)
  path.unlink()


def test_load_validation_report_dict():
  """Loads validation report dictionary."""
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    json.dump({"DummyOpNonExistent": "Pass"}, f)
    path = Path(f.name)
  manager.load_validation_report(path)
  path.unlink()
  assert manager.is_verified("DummyOpNonExistent") == "Pass"


def test_is_verified_fallback():
  """Checks if is verified fallback."""
  manager = SemanticsManager()
  assert manager.is_verified("MissingOp") is True


def test_get_all_rng_methods():
  """Gets all rng methods."""
  manager = SemanticsManager()
  assert isinstance(manager.get_all_rng_methods(), set)


def test_get_patterns():
  """Gets patterns."""
  manager = SemanticsManager()
  assert isinstance(manager.get_patterns(), list)


def test_update_definition_inject_fields_and_validation_error():
  """Updates definition inject fields and validation correctly handling an error."""
  manager = SemanticsManager()
  manager.update_definition("DummyOpNonExistent", {})
  manager.update_definition("DummyOpNonExistent", {"std_args": "invalid_type"})


def test_update_definition_extras_tier_and_corrupt_file_and_variants(monkeypatch):
  """Updates definition extras tier and corrupt file and variants."""
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
  """Updates definition write correctly handling an exception."""
  manager = SemanticsManager()
  import ml_switcheroo.semantics.manager as sm

  with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    monkeypatch.setattr(sm, "resolve_semantics_dir", lambda: tmppath)
    original_open = open

    def mock_open_write(*args, **kwargs):
      """Provides a mock open write for testing."""
      if "w" in args[1]:
        raise PermissionError("Cannot write")
      return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_write):
      manager.update_definition(
        "DummyOpNonExistent", {"operation": "DummyOpNonExistent", "description": "bar", "std_args": [], "variants": {}}
      )


def test_get_definition():
  """Gets definition."""
  manager = SemanticsManager()
  manager.data["MyOp"] = {"operation": "MyOp"}
  manager._reverse_index["tf.myop"] = ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("tf.myop") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("MyOp") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("NonExistent") is None
