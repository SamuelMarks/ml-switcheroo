"""Test suite for the Convert module."""

import pytest
import tempfile
from unittest import mock
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.convert import handle_convert, _convert_single_file, _print_batch_summary
from ml_switcheroo.core.engine import ConversionResult


def test_handle_convert_infer_source(monkeypatch):
  """Tests inferring the source framework from file extension."""
  with tempfile.TemporaryDirectory() as tmp:
    in_file = Path(tmp) / "input.mlir"
    in_file.write_text("module {}")
    out_file = Path(tmp) / "output.py"

    import ml_switcheroo.cli.handlers.convert

    mock_load = mock.MagicMock(return_value=0)
    monkeypatch.setattr(ml_switcheroo.cli.handlers.convert, "load_plugins", mock_load)

    # We need to mock the entire execution basically, so just mock process_file
    with mock.patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_proc:
      # Also mock RuntimeConfig.load
      with mock.patch("ml_switcheroo.config.RuntimeConfig.load") as mock_conf:
        mock_conf.return_value = mock.MagicMock(plugin_paths=None, strict=False, source="mlir", target="jax")
        handle_convert(
          in_file,
          out_file,
          source=None,
          target="jax",
          verify=False,
          strict=None,
          intermediate=None,
          plugin_settings={},
          json_trace_path=None,
          enable_sharding=False,
        )
        mock_proc.assert_called_once()
        mock_conf.assert_called_once()
        assert mock_conf.call_args[1]["source"] == "mlir"

    # Test unknown extension inference failure
    in_file2 = Path(tmp) / "input.unknown"
    in_file2.write_text("module {}")
    with mock.patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_proc:
      with mock.patch("ml_switcheroo.config.RuntimeConfig.load") as mock_conf:
        mock_conf.return_value = mock.MagicMock(plugin_paths=None, strict=False, source=None, target="jax")
        handle_convert(
          in_file2,
          out_file,
          source=None,
          target="jax",
          verify=False,
          strict=None,
          intermediate=None,
          plugin_settings={},
          json_trace_path=None,
          enable_sharding=False,
        )
        assert mock_conf.call_args[1]["source"] is None


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


def test_handle_convert_plugins_and_batch(monkeypatch, tmp_path):
  """Tests loading external plugins and processing a directory."""
  in_dir = tmp_path / "src"
  in_dir.mkdir()
  (in_dir / "a.py").write_text("import torch")
  out_dir = tmp_path / "out"

  from ml_switcheroo.cli.handlers.convert import handle_convert
  import ml_switcheroo.cli.handlers.convert

  mock_load = mock.MagicMock(return_value=1)  # simulating 1 plugin loaded
  monkeypatch.setattr(ml_switcheroo.cli.handlers.convert, "load_plugins", mock_load)

  with mock.patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_convert:
    mock_convert.return_value = ConversionResult(success=True, code="source", errors=[])

    with mock.patch("ml_switcheroo.config.RuntimeConfig.load") as mock_conf:
      mock_conf.return_value = mock.MagicMock(plugin_paths=["/my/plugins"], strict=False, source="torch", target="jax")

      # Test batch directory conversion with a JSON trace path specified
      handle_convert(
        in_dir,
        out_dir,
        source=None,
        target="jax",
        verify=False,
        strict=None,
        intermediate=None,
        plugin_settings={},
        json_trace_path=Path("trace.json"),
        enable_sharding=False,
      )

      mock_load.assert_called_once_with(extra_dirs=["/my/plugins"])
      mock_convert.assert_called_once()
      assert mock_convert.call_args[0][0].name == "a.py"
      assert mock_convert.call_args[0][1].name == "a.py"
      assert mock_convert.call_args[0][5].name == "a.trace.json"


def test_print_batch_summary_warnings():
  """Tests the batch summary output for warnings and errors."""
  from ml_switcheroo.cli.handlers.convert import _print_batch_summary

  results = {
    "file1.py": ConversionResult(success=True, code="code", errors=[]),  # success
    "file2.py": ConversionResult(success=True, code="code", errors=["Warning!"]),  # warnings
    "file3.py": ConversionResult(success=False, code="", errors=["Error!"]),  # error
  }
  with mock.patch("ml_switcheroo.cli.handlers.convert.console.print") as mock_print:
    _print_batch_summary(results)
    # The table is printed, just verify it's called
    assert mock_print.call_count >= 1


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
