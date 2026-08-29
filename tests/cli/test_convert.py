"""Test module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.cli.handlers.convert import _convert_single_file, _print_batch_summary, handle_convert
from ml_switcheroo.core.engine import ConversionResult


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
@patch("ml_switcheroo.cli.handlers.convert.load_plugins")
def test_handle_convert_single_file(mock_load_plugins: MagicMock, mock_convert_single: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_convert_single.return_value = ConversionResult(success=True, code="out", errors=[])

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("code")
  out_file: Path = tmp_path / "out.py"

  res: int = handle_convert(
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
def test_handle_convert_dir(mock_convert_single: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_convert_single.return_value = ConversionResult(success=True, code="out", errors=[])

  in_dir: Path = tmp_path / "src"
  in_dir.mkdir()
  (in_dir / "f1.py").write_text("code1")
  (in_dir / "f2.py").write_text("code2")

  out_dir: Path = tmp_path / "out"

  res: int = handle_convert(
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


def test_handle_convert_not_found() -> None:
  """Docstring."""
  res: int = handle_convert(Path("nonexistent.py"), None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_dir_no_out(tmp_path: Path) -> None:
  """Docstring."""
  in_dir: Path = tmp_path / "src"
  in_dir.mkdir()
  res: int = handle_convert(in_dir, None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_infer_source(tmp_path: Path) -> None:
  """Docstring."""
  in_file: Path = tmp_path / "in.sass"
  in_file.write_text("code")
  m: MagicMock
  with patch("ml_switcheroo.cli.handlers.convert._convert_single_file") as m:
    m.return_value = ConversionResult(success=True, code="")
    handle_convert(in_file, None, None, None, False, False, None, {})
    config: MagicMock = m.call_args[0][4]
    assert config.source_framework == "sass"


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_success(mock_engine_class: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_engine: MagicMock = mock_engine_class.return_value
  mock_engine.run.return_value = ConversionResult(success=True, code="out code", trace_events=[{"event": "start"}])

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("in code")
  out_file: Path = tmp_path / "out.py"
  trace_file: Path = tmp_path / "trace.json"

  res: ConversionResult = _convert_single_file(in_file, out_file, MagicMock(), False, MagicMock(), trace_file)

  assert res.success is True
  assert out_file.read_text() == "out code"
  assert json.loads(trace_file.read_text()) == [{"event": "start"}]


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_stdout(
  mock_engine_class: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Docstring."""
  mock_engine: MagicMock = mock_engine_class.return_value
  mock_engine.run.return_value = ConversionResult(success=True, code="out code stdout")

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("in")

  _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)

  captured = capsys.readouterr()
  assert "out code stdout" in captured.out


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_exception(mock_engine_class: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_engine: MagicMock = mock_engine_class.return_value
  mock_engine.run.side_effect = ValueError("Boom")

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("in")

  res: ConversionResult = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)
  assert res.success is False
  assert res.errors is not None
  assert "Boom" in res.errors[0]


@patch("ml_switcheroo.cli.handlers.convert.subprocess.run")
@patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator")
@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_verify(
  mock_engine: MagicMock, mock_harness: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
  """Docstring."""
  mock_engine.return_value.run.return_value = ConversionResult(success=True, code="out")
  mock_run.return_value.returncode = 0

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("in")
  out_file: Path = tmp_path / "out.py"

  config: MagicMock = MagicMock()
  config.source_framework = "torch"
  config.target_framework = "jax"

  _convert_single_file(in_file, out_file, MagicMock(), True, config, None)

  mock_harness.return_value.generate.assert_called_once()
  mock_run.assert_called_once()


def test_print_batch_summary(capsys: pytest.CaptureFixture[str]) -> None:
  """Docstring."""
  results: dict[str, ConversionResult] = {
    "ok.py": ConversionResult(success=True, code=""),
    "fail.py": ConversionResult(success=False, code="", errors=["Bad"]),
    "warn.py": ConversionResult(success=True, code="", errors=["Warn"]),
  }

  _print_batch_summary(results)

  captured = capsys.readouterr()
  assert "1 Passed" in captured.out
  assert "2 with Issues" in captured.out


def test_print_batch_summary_perfect(capsys: pytest.CaptureFixture[str]) -> None:
  """Docstring."""
  results: dict[str, ConversionResult] = {"ok.py": ConversionResult(success=True, code="")}
  _print_batch_summary(results)
  captured = capsys.readouterr()
  assert "converted perfectly" in captured.out


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
@patch("ml_switcheroo.cli.handlers.convert.load_plugins")
def test_handle_convert_plugins(mock_load: MagicMock, mock_conv: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_conv.return_value = ConversionResult(success=True, code="")
  mock_load.return_value = 1

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("code")

  mock_load_config: MagicMock
  with patch("ml_switcheroo.cli.handlers.convert.RuntimeConfig.load") as mock_load_config:
    config: MagicMock = MagicMock()
    config.plugin_paths = [Path("some/path")]
    mock_load_config.return_value = config

    handle_convert(in_file, None, None, None, False, False, None, {})
    mock_load.assert_called_once()


@patch("ml_switcheroo.cli.handlers.convert._convert_single_file")
def test_handle_convert_single_fail_fast(mock_conv: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_conv.return_value = ConversionResult(success=False, code="", errors=["err"])
  in_file: Path = tmp_path / "in.py"
  in_file.write_text("code")

  res: int = handle_convert(in_file, None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_dir_empty(tmp_path: Path) -> None:
  """Docstring."""
  in_dir: Path = tmp_path / "src"
  in_dir.mkdir()

  res: int = handle_convert(in_dir, tmp_path / "out", None, None, False, False, None, {})
  assert res == 0


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_trace_exception(mock_engine: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_engine.return_value.run.return_value = ConversionResult(
    success=True, code="out", trace_events=[{"event": "start"}]
  )

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("code")

  # Provide a bad path for trace to trigger exception (like a directory)
  trace_path: Path = tmp_path / "bad"
  trace_path.mkdir()

  res: ConversionResult = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), trace_path)
  assert res.success is True


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_fail(mock_engine: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  mock_engine.return_value.run.return_value = ConversionResult(success=False, code="out")
  in_file: Path = tmp_path / "in.py"
  in_file.write_text("code")

  res: ConversionResult = _convert_single_file(in_file, None, MagicMock(), False, MagicMock(), None)
  assert res.success is False


@patch("ml_switcheroo.cli.handlers.convert.subprocess.run")
@patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator")
@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_convert_single_file_verify_fail(
  mock_engine: MagicMock, mock_harness: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
  """Docstring."""
  mock_engine.return_value.run.return_value = ConversionResult(success=True, code="out")
  mock_run.return_value.returncode = 1

  in_file: Path = tmp_path / "in.py"
  in_file.write_text("in")

  config: MagicMock = MagicMock()

  # We also want to hit line 178 (verify and not effective_out)
  res: ConversionResult = _convert_single_file(in_file, None, MagicMock(), True, config, None)

  assert res.success is True
  assert res.errors is not None
  assert "Verification Harness Failed" in res.errors
