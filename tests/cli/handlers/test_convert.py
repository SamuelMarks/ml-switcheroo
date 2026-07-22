"""Test suite for the Convert module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.convert import handle_convert, _convert_single_file, _print_batch_summary
from ml_switcheroo.core.engine import ConversionResult


@pytest.fixture
def mock_config():
  """Provides a mock configuration for testing."""
  with patch("ml_switcheroo.config.RuntimeConfig.load") as mock_load:
    mock_conf = MagicMock()
    mock_conf.plugin_paths = []
    mock_conf.source_framework = "torch"
    mock_conf.target_framework = "jax"
    mock_load.return_value = mock_conf
    yield mock_load


@pytest.fixture
def mock_engine():
  """Provides a mock engine for testing."""
  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock:
    yield mock


def test_handle_convert_input_not_found(mock_config):
  """Handles convert input not found."""
  assert handle_convert(Path("nonexistent.py"), None, None, None, False, None, None, {}) == 1


def test_handle_convert_single_file_success(mock_config, mock_engine, tmp_path):
  """Handles convert a single file successfully."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  assert handle_convert(input_file, None, None, None, False, None, None, {}) == 0


def test_handle_convert_single_file_with_output(mock_config, mock_engine, tmp_path):
  """Handles convert a single file with output."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  out_file = tmp_path / "out.py"
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  assert handle_convert(input_file, out_file, None, None, False, None, None, {}) == 0
  assert out_file.exists()
  assert out_file.read_text() == "y = 1"


def test_handle_convert_single_file_failure(mock_config, mock_engine, tmp_path):
  """Handles convert a single file successfully handling failure."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=False, errors=["some error"])
  assert handle_convert(input_file, None, None, None, False, None, None, {}) == 1


def test_handle_convert_single_file_failure_exit(mock_config, mock_engine, tmp_path):
  """Handles convert a single file successfully handling failure exit."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=False, errors=["some error"])
  assert handle_convert(input_file, None, None, None, False, None, None, {}) == 1


def test_handle_convert_dir_no_out(mock_config, tmp_path):
  """Handles convert a directory no output."""
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  assert handle_convert(input_dir, None, None, None, False, None, None, {}) == 1


def test_handle_convert_dir_empty(mock_config, tmp_path):
  """Handles convert a directory empty."""
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  out_dir = tmp_path / "out"
  assert handle_convert(input_dir, out_dir, None, None, False, None, None, {}) == 0


def test_handle_convert_dir_success(mock_config, mock_engine, tmp_path):
  """Handles convert a directory successfully."""
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  (input_dir / "in.py").write_text("x = 1")
  out_dir = tmp_path / "out"
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  assert handle_convert(input_dir, out_dir, None, None, False, None, None, {}, json_trace_path=Path("dummy")) == 0
  assert (out_dir / "in.py").exists()


def test_convert_single_file_exception(mock_config, tmp_path):
  """Converts a single file correctly handling an exception."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine", side_effect=ValueError("boom")):
    res = _convert_single_file(input_file, None, MagicMock(), False, MagicMock())
    assert res.success is False
    assert "boom" in res.errors[0]


def test_convert_single_file_json_trace(mock_config, mock_engine, tmp_path):
  """Converts a single file JSON trace."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  trace_file = tmp_path / "trace.json"
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1", trace_events=[{"event": 1}])
  _convert_single_file(input_file, None, MagicMock(), False, MagicMock(), json_trace_path=trace_file)
  assert trace_file.exists()


def test_convert_single_file_json_trace_error(mock_config, mock_engine, tmp_path):
  """Converts a single file JSON trace correctly handling an error."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  trace_file = tmp_path / "ro" / "trace.json"
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1", trace_events=[{"event": 1}])
  with patch("pathlib.Path.mkdir", side_effect=Exception("boom")):
    _convert_single_file(input_file, None, MagicMock(), False, MagicMock(), json_trace_path=trace_file)


def test_convert_single_file_verify_success(mock_config, mock_engine, tmp_path):
  """Converts a single file verify successfully."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  with patch("subprocess.run") as mock_subp:
    mock_subp.return_value.returncode = 0
    res = _convert_single_file(input_file, None, MagicMock(), True, mock_config.return_value)
    assert res.success is True


def test_convert_single_file_verify_failure(mock_config, mock_engine, tmp_path):
  """Converts a single file verify successfully handling failure."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  with patch("subprocess.run") as mock_subp:
    mock_subp.return_value.returncode = 1
    res = _convert_single_file(input_file, None, MagicMock(), True, mock_config.return_value)
    assert "Verification Harness Failed" in res.errors


def test_print_batch_summary():
  """Prints batch summary."""
  _print_batch_summary({"a.py": ConversionResult(success=True)})
  _print_batch_summary({"a.py": ConversionResult(success=False, errors=["err"])})


def test_load_plugins(mock_config, mock_engine, tmp_path):
  """Loads plugins."""
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  mock_config.return_value.plugin_paths = ["some/path"]
  mock_instance = mock_engine.return_value
  mock_instance.run.return_value = ConversionResult(success=True, code="y = 1")
  with patch("ml_switcheroo.cli.handlers.convert.load_plugins", return_value=1) as mock_load:
    handle_convert(input_file, None, None, None, False, None, None, {})
    mock_load.assert_called_once()
