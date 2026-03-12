import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.cli.handlers.convert import handle_convert, _convert_single_file, _print_batch_summary
from ml_switcheroo.core.engine import ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


def test_handle_convert_input_not_found(tmp_path):
  res = handle_convert(tmp_path / "missing.py", None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_single_file(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  out_file = tmp_path / "out.py"

  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_conv:
    mock_conv.return_value = ConversionResult(success=True, code="y = 1", errors=[])
    res = handle_convert(input_file, out_file, None, None, False, False, None, {})
    assert res == 0
    mock_conv.assert_called_once()


def test_handle_convert_single_file_fails(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_conv:
    mock_conv.return_value = ConversionResult(success=False, code="", errors=["fail"])
    res = handle_convert(input_file, None, None, None, False, False, None, {})
    assert res == 1


def test_handle_convert_dir_no_output(tmp_path):
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  res = handle_convert(input_dir, None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_dir_empty(tmp_path):
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  out_dir = tmp_path / "out"
  res = handle_convert(input_dir, out_dir, None, None, False, False, None, {})
  assert res == 0


def test_handle_convert_dir_with_files(tmp_path):
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  (input_dir / "test1.py").write_text("a=1")
  out_dir = tmp_path / "out"

  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_conv:
    mock_conv.return_value = ConversionResult(success=True, code="", errors=[])
    res = handle_convert(input_dir, out_dir, None, None, False, False, None, {})
    assert res == 0


def test_handle_convert_dir_with_files_and_trace(tmp_path):
  input_dir = tmp_path / "src"
  input_dir.mkdir()
  (input_dir / "test1.py").write_text("a=1")
  out_dir = tmp_path / "out"

  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_conv:
    mock_conv.return_value = ConversionResult(success=True, code="", errors=[])
    res = handle_convert(input_dir, out_dir, None, None, False, False, None, {}, json_trace_path=tmp_path / "trace.json")
    assert res == 0


def test_handle_convert_with_plugins(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  class FakeCfg:
    plugin_paths = ["a/b"]
    source_framework = "torch"
    target_framework = "jax"

  with (
    patch("ml_switcheroo.cli.handlers.convert.RuntimeConfig.load") as mock_cfg,
    patch("ml_switcheroo.cli.handlers.convert.load_plugins") as mock_load,
    patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as mock_conv,
  ):
    mock_cfg.return_value = FakeCfg()
    mock_load.return_value = 1
    mock_conv.return_value = ConversionResult(success=True, code="", errors=[])
    res = handle_convert(input_file, None, None, None, False, False, None, {})
    assert res == 0


def test_convert_single_file(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  out_file = tmp_path / "out.py"
  trace_file = tmp_path / "trace.json"

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine:
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=True, code="y=2", errors=[], trace_events=[{"event": 1}])
    mock_engine.return_value = mock_instance

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, out_file, SemanticsManager(), False, cfg, json_trace_path=trace_file)

    assert res.success
    assert out_file.read_text() == "y=2"
    assert trace_file.exists()


def test_convert_single_file_trace_failure(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")
  out_file = tmp_path / "out.py"
  # Provide a directory path to trace_file so open() fails to simulate error writing trace
  trace_file = tmp_path / "dir_trace"
  trace_file.mkdir()

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine:
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=True, code="y=2", errors=[], trace_events=[{"event": 1}])
    mock_engine.return_value = mock_instance

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, out_file, SemanticsManager(), False, cfg, json_trace_path=trace_file)

    assert res.success
    assert out_file.read_text() == "y=2"


def test_convert_single_file_engine_fails(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine:
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=False, code="", errors=["fail"])
    mock_engine.return_value = mock_instance

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, None, SemanticsManager(), False, cfg, None)
    assert not res.success


def test_convert_single_file_no_output(tmp_path, capsys):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine:
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=True, code="print('hi')", errors=[])
    mock_engine.return_value = mock_instance

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, None, SemanticsManager(), False, cfg, None)
    assert res.success
    out, _ = capsys.readouterr()
    assert "print('hi')" in out


def test_convert_single_file_verify_success(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with (
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine,
    patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator") as mock_harness,
    patch("ml_switcheroo.cli.handlers.convert.subprocess.run") as mock_run,
  ):
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=True, code="y=1", errors=[])
    mock_engine.return_value = mock_instance

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, None, SemanticsManager(), True, cfg, None)
    assert res.success


def test_convert_single_file_verify_fails(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with (
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine,
    patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator") as mock_harness,
    patch("ml_switcheroo.cli.handlers.convert.subprocess.run") as mock_run,
  ):
    mock_instance = MagicMock()
    mock_instance.run.return_value = ConversionResult(success=True, code="y=1", errors=[])
    mock_engine.return_value = mock_instance

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_run.return_value = mock_proc

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, None, SemanticsManager(), True, cfg, None)
    assert res.success
    assert "Verification Harness Failed" in res.errors


def test_convert_single_file_exception(tmp_path):
  input_file = tmp_path / "in.py"
  input_file.write_text("x = 1")

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_engine:
    mock_engine.side_effect = ValueError("boom")

    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    res = _convert_single_file(input_file, None, SemanticsManager(), False, cfg, None)
    assert not res.success
    assert "boom" in res.errors[0]


def test_print_batch_summary_all_success(capsys):
  results = {"a.py": ConversionResult(success=True, code="", errors=[])}

  from rich.console import Console

  with patch("ml_switcheroo.cli.handlers.convert.console", Console(force_terminal=False)):
    _print_batch_summary(results)


def test_print_batch_summary_mixed(capsys):
  results = {
    "a.py": ConversionResult(success=True, code="", errors=[]),
    "b.py": ConversionResult(success=False, code="", errors=["err"]),
    "c.py": ConversionResult(success=True, code="", errors=["warn"]),
  }

  from rich.console import Console

  with patch("ml_switcheroo.cli.handlers.convert.console", Console(force_terminal=False)):
    _print_batch_summary(results)
