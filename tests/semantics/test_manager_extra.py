import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


def test_load_validation_report_not_found(capsys):
  manager = SemanticsManager()
  manager.load_validation_report(Path("/non_existent_file.json"))
  out, err = capsys.readouterr()
  assert "Validation report not found" in out


def test_load_validation_report_exception(capsys):
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    path = Path(f.name)
  with patch("builtins.open", side_effect=Exception("mocked error")):
    manager.load_validation_report(path)
  path.unlink()


def test_load_validation_report_dict(capsys):
  manager = SemanticsManager()
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    json.dump({"DummyOpNonExistent": "Pass"}, f)
    path = Path(f.name)
  manager.load_validation_report(path)
  path.unlink()
  out, err = capsys.readouterr()
  assert "Loaded 1 verification statuses" in out
  assert manager.is_verified("DummyOpNonExistent") == "Pass"


def test_is_verified_fallback():
  manager = SemanticsManager()
  # If not tracked, defaults to True
  assert manager.is_verified("MissingOp") is True


def test_get_all_rng_methods():
  manager = SemanticsManager()
  assert isinstance(manager.get_all_rng_methods(), set)


def test_get_patterns():
  manager = SemanticsManager()
  assert isinstance(manager.get_patterns(), list)


def test_update_definition_inject_fields_and_validation_error(capsys):
  manager = SemanticsManager()
  manager.update_definition("DummyOpNonExistent", {})
  manager.update_definition("DummyOpNonExistent", {"std_args": "invalid_type"})
  # This might actually pass due to the Schema default fields or print validation error
  out, err = capsys.readouterr()
  assert "Cannot update invalid definition" in out


def test_update_definition_extras_tier_and_corrupt_file_and_variants(monkeypatch, capsys):
  manager = SemanticsManager()
  manager._key_origins["MyExtraOp"] = SemanticTier.EXTRAS.value

  import ml_switcheroo.semantics.manager as sm

  with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    monkeypatch.setattr(sm, "resolve_semantics_dir", lambda: tmppath)

    file_path = tmppath / "k_framework_extras.json"
    with open(file_path, "w") as f:
      f.write("corrupt json")

    manager.update_definition(
      "MyExtraOp",
      {
        "operation": "MyExtraOp",
        "description": "foo",
        "std_args": [],
        "variants": {"tensorflow": {"api": "tf.my_extra_op"}},
      },
    )

    with open(file_path, "r") as f:
      content = json.load(f)
      assert "MyExtraOp" in content
      assert content["MyExtraOp"]["operation"] == "MyExtraOp"


def test_update_definition_write_exception(monkeypatch, capsys):
  manager = SemanticsManager()
  import ml_switcheroo.semantics.manager as sm

  with tempfile.TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    monkeypatch.setattr(sm, "resolve_semantics_dir", lambda: tmppath)

    original_open = open

    def mock_open_write(*args, **kwargs):
      if "w" in args[1]:
        raise PermissionError("Cannot write")
      return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_write):
      manager.update_definition(
        "DummyOpNonExistent", {"operation": "DummyOpNonExistent", "description": "bar", "std_args": [], "variants": {}}
      )

    out, err = capsys.readouterr()
    assert "Failed to write update" in out


def test_get_definition():
  manager = SemanticsManager()
  manager.data["MyOp"] = {"operation": "MyOp"}
  manager._reverse_index["tf.myop"] = ("MyOp", manager.data["MyOp"])

  assert manager.get_definition("tf.myop") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("MyOp") == ("MyOp", manager.data["MyOp"])
  assert manager.get_definition("NonExistent") is None
