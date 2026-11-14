"""Unit tests for the `define` CLI command handler (`ml_switcheroo.cli.handlers.define`)."""

from pathlib import Path

import pytest
import yaml

import ml_switcheroo.cli.handlers.define as define_mod
from ml_switcheroo.cli.handlers.define import handle_define


def test_handle_define_missing_malformed_invalid_and_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `handle_define` rejects missing, malformed, and schema-invalid files and copies valid ODL specs.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert handle_define(tmp_path / "does_not_exist.yaml") == 1

  bad_yaml = tmp_path / "malformed.yaml"
  bad_yaml.write_text("invalid: yaml: [}", encoding="utf-8")
  assert handle_define(bad_yaml) == 1

  bad_schema = tmp_path / "invalid_schema.yaml"
  bad_schema.write_text("description: Missing required operation field", encoding="utf-8")
  assert handle_define(bad_schema) == 1

  dest_dir = tmp_path / "semantics"
  monkeypatch.setattr(define_mod, "resolve_semantics_dir", lambda: dest_dir)

  valid_yaml = yaml.dump({"operation": "MyOp", "description": "Test op", "variants": {}})
  source_path = tmp_path / "source.yaml"
  source_path.write_text(valid_yaml, encoding="utf-8")

  assert handle_define(source_path) == 0
  target_path = dest_dir / "odl" / "MyOp.yaml"
  assert target_path.exists()
  assert target_path.read_text(encoding="utf-8") == valid_yaml
