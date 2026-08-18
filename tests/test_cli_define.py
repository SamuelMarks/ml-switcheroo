"""Tests for the define CLI command handler."""

from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from ml_switcheroo.cli.handlers.define import handle_define


@pytest.fixture
def valid_yaml_content() -> str:
  """Returns valid YAML content for an OperationDef.

  Returns:
      str: The YAML string.
  """
  return yaml.dump({"operation": "MyOp", "description": "Test op", "variants": {}})


def test_handle_define_file_not_found(tmp_path: Path) -> None:
  """Test that define fails if file does not exist.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  fake_path = tmp_path / "does_not_exist.yaml"
  assert handle_define(fake_path) == 1


def test_handle_define_invalid_yaml(tmp_path: Path) -> None:
  """Test that define fails if YAML is malformed.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  fake_path = tmp_path / "invalid.yaml"
  fake_path.write_text("operation: 123\\ninvalid_field: [}")  # Bad YAML syntax

  with patch("ml_switcheroo.cli.handlers.define.log_error") as mock_log:
    assert handle_define(fake_path) == 1
    mock_log.assert_called_once()


def test_handle_define_validation_error(tmp_path: Path) -> None:
  """Test that define fails if it fails Pydantic validation.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  fake_path = tmp_path / "invalid_schema.yaml"
  # operation is missing, which is required
  fake_path.write_text("description: Test op")

  with patch("ml_switcheroo.cli.handlers.define.log_error") as mock_log:
    assert handle_define(fake_path) == 1
    mock_log.assert_called_once()


def test_handle_define_success(tmp_path: Path, valid_yaml_content: str) -> None:
  """Test successful validation and copying.

  Args:
      tmp_path: Pytest temporary directory fixture.
      valid_yaml_content: Fixture providing valid YAML.
  """
  source_path = tmp_path / "source.yaml"
  source_path.write_text(valid_yaml_content)

  dest_dir = tmp_path / "semantics"

  with patch("ml_switcheroo.cli.handlers.define.resolve_semantics_dir", return_value=dest_dir):
    with patch("ml_switcheroo.cli.handlers.define.log_success") as mock_success:
      result = handle_define(source_path)
      assert result == 0

      target_path = dest_dir / "odl" / "MyOp.yaml"
      assert target_path.exists()
      assert target_path.read_text() == valid_yaml_content
      mock_success.assert_called_once()
