"""Test module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.convert import handle_convert, _convert_single_file, _print_batch_summary
from ml_switcheroo.core.engine import ConversionResult


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
@patch("ml_switcheroo.cli.handlers.convert.load_plugins")
def test_handle_convert_single_file(mock_load_plugins, mock_convert_single, tmp_path):
  """Test element."""
  mock_convert_single.return_value = ConversionResult(success=True, code="out", errors=[])

  in_file = tmp_path / "in.py"
  in_file.write_text("code")
  out_file = tmp_path / "out.py"

  res = handle_convert(
    input_path=in_file,
    output_path=out_file,
    source="torch",
    target="jax",
    verify=False,
    strict=False,
    intermediate=None,
    plugin_settings={},
    json_trace_path=None,
    enable_sharding=False,
  )

  assert res == 0
  mock_convert_single.assert_called_once()


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
def test_handle_convert_dir(mock_convert_single, tmp_path):
  """Test element."""
  mock_convert_single.return_value = ConversionResult(success=True, code="out", errors=[])

  in_dir = tmp_path / "src"
  in_dir.mkdir()
  (in_dir / "f1.py").write_text("code1")
  (in_dir / "f2.py").write_text("code2")

  out_dir = tmp_path / "out"

  res = handle_convert(
    input_path=in_dir,
    output_path=out_dir,
    source=None,
    target=None,
    verify=False,
    strict=False,
    intermediate=None,
    plugin_settings={},
    json_trace_path=tmp_path / "trace.json",  # will cause batch trace to be used
    enable_sharding=False,
  )

  assert res == 0
  assert mock_convert_single.call_count == 2


def test_handle_convert_not_found():
  """Test element."""
  res = handle_convert(Path("nonexistent.py"), None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_dir_no_out(tmp_path):
  """Test element."""
  in_dir = tmp_path / "src"
  in_dir.mkdir()
  res = handle_convert(in_dir, None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_infer_source(tmp_path):
  """Test element."""
  in_file = tmp_path / "in.sass"
  in_file.write_text("code")
  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as m:
    m.return_value = ConversionResult(success=True, code="")
    handle_convert(in_file, None, None, None, False, False, None, {})
    config = m.call_args[0][4]
    assert config.source_framework == "sass"


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_success(mock_engine_class, tmp_path):
  """Test element."""
  mock_engine = mock_engine_class.return_value
  mock_engine.run.return_value = ConversionResult(success=True, code="out code", trace_events=[{"event": "start"}])

  in_file = tmp_path / "in.py"
  in_file.write_text("in code")
  out_file = tmp_path / "out.py"
  trace_file = tmp_path / "trace.json"

  res = _convert_single_file(in_file, out_file, MagicMock(), False, MagicMock(), trace_file)

  assert res.success is True
  assert out_file.read_text() == "out code"
  assert json.loads(trace_file.read_text()) == [{"event": "start"}]


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_stdout(mock_engine_class, tmp_path, capsys):
  """Test element."""
  mock_engine = mock_engine_class.return_value
  mock_engine.run.return_value = ConversionResult(success=True, code="out code stdout")

  in_file = tmp_path / "in.py"
  in_file.write_text("in")

  _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)

  captured = capsys.readouterr()
  assert "out code stdout" in captured.out


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_exception(mock_engine_class, tmp_path):
  """Test element."""
  mock_engine = mock_engine_class.return_value
  mock_engine.run.side_effect = ValueError("Boom")

  in_file = tmp_path / "in.py"
  in_file.write_text("in")

  res = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)
  assert res.success is False
  assert "Boom" in res.errors[0]


@patch("ml_switcheroo.cli.handlers.convert.subprocess.run")
@patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator")
@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_verify(mock_engine, mock_harness, mock_run, tmp_path):
  """Test element."""
  mock_engine.return_value.run.return_value = ConversionResult(success=True, code="out")
  mock_run.return_value.returncode = 0

  in_file = tmp_path / "in.py"
  in_file.write_text("in")
  out_file = tmp_path / "out.py"

  config = MagicMock()
  config.source_framework = "torch"
  config.target_framework = "jax"

  _convert_single_file(in_file, out_file, MagicMock(), True, config, None)

  mock_harness.return_value.generate.assert_called_once()
  mock_run.assert_called_once()


def test_print_batch_summary(capsys):
  """Test element."""
  results = {
    "ok.py": ConversionResult(success=True, code=""),
    "fail.py": ConversionResult(success=False, code="", errors=["Bad"]),
    "warn.py": ConversionResult(success=True, code="", errors=["Warn"]),
  }

  _print_batch_summary(results)

  captured = capsys.readouterr()
  assert "1 Passed" in captured.out
  assert "2 with Issues" in captured.out


def test_print_batch_summary_perfect(capsys):
  """Test element."""
  results = {"ok.py": ConversionResult(success=True, code="")}
  _print_batch_summary(results)
  captured = capsys.readouterr()
  assert "converted perfectly" in captured.out


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
@patch("ml_switcheroo.cli.handlers.convert.load_plugins")
def test_handle_convert_plugins(mock_load, mock_conv, tmp_path):
  """Test element."""
  mock_conv.return_value = ConversionResult(success=True, code="")
  mock_load.return_value = 1

  in_file = tmp_path / "in.py"
  in_file.write_text("code")

  # Needs a config mock to return a config with plugin_paths
  with patch("ml_switcheroo.cli.handlers.convert.RuntimeConfig.load") as mock_load_config:
    config = MagicMock()
    config.plugin_paths = [Path("some/path")]
    mock_load_config.return_value = config

    handle_convert(in_file, None, None, None, False, False, None, {})
    mock_load.assert_called_once()


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
def test_handle_convert_single_fail_fast(mock_conv, tmp_path):
  """Test element."""
  mock_conv.return_value = ConversionResult(success=False, code="", errors=["err"])
  in_file = tmp_path / "in.py"
  in_file.write_text("code")

  res = handle_convert(in_file, None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_dir_empty(tmp_path):
  """Test element."""
  in_dir = tmp_path / "src"
  in_dir.mkdir()

  res = handle_convert(in_dir, tmp_path / "out", None, None, False, False, None, {})
  assert res == 0


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_trace_exception(mock_engine, tmp_path):
  """Test element."""
  mock_engine.return_value.run.return_value = ConversionResult(
    success=True, code="out", trace_events=[{"event": "start"}]
  )

  in_file = tmp_path / "in.py"
  in_file.write_text("code")

  # Provide a bad path for trace to trigger exception (like a directory)
  trace_path = tmp_path / "bad"
  trace_path.mkdir()

  res = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), trace_path)
  assert res.success is True


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_fail(mock_engine, tmp_path):
  """Test element."""
  mock_engine.return_value.run.return_value = ConversionResult(success=False, code="out")
  in_file = tmp_path / "in.py"
  in_file.write_text("code")

  res = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)
  assert res.success is False


@patch("ml_switcheroo.cli.handlers.convert.subprocess.run")
@patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator")
@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_verify_fail(mock_engine, mock_harness, mock_run, tmp_path):
  """Test element."""
  mock_engine.return_value.run.return_value = ConversionResult(success=True, code="out")
  mock_run.return_value.returncode = 1

  in_file = tmp_path / "in.py"
  in_file.write_text("in")

  config = MagicMock()

  # We also want to hit line 178 (verify and not effective_out)
  res = _convert_single_file(in_file, None, MagicMock(), True, config, None)

  assert res.success is True
  assert "Verification Harness Failed" in res.errors
