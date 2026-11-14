"""Unit tests for the `convert` CLI handler (`ml_switcheroo.cli.handlers.convert`)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo.cli.handlers.convert import (
  CliConvertOptions,
  _convert_single_file,
  _print_batch_summary,
  handle_convert,
)
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


def test_handle_convert_real_files_directories_and_source_inference(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verify `handle_convert` transpiles real single files, directories, stdout, JSON traces, and extensions.

  Args:
      tmp_path: Pytest temporary directory fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  assert handle_convert(CliConvertOptions(input_path=tmp_path / "missing.py", strict=False, plugin_settings={})) == 1

  # Single file Torch -> JAX conversion writing output and JSON trace
  in_py = tmp_path / "model.py"
  in_py.write_text(f"import torch{_NL}y = torch.abs(x){_NL}", encoding="utf-8")
  out_py = tmp_path / "out.py"
  trace_json = tmp_path / "trace.json"

  rc = handle_convert(
    CliConvertOptions(
      input_path=in_py,
      output_path=out_py,
      source="torch",
      target="jax",
      verify=False,
      strict=False,
      intermediate=None,
      plugin_settings={},
      json_trace_path=trace_json,
    )
  )
  assert rc == 0
  assert "jnp.abs(x)" in out_py.read_text(encoding="utf-8")
  assert isinstance(json.loads(trace_json.read_text(encoding="utf-8")), list)

  # Single file conversion printed to stdout when `output_path=None`
  assert handle_convert(CliConvertOptions(input_path=in_py, source="torch", target="jax", strict=False)) == 0
  assert "jnp.abs(x)" in capsys.readouterr().out

  # Extension-based source inference (`.mlir` and `.nvidia_sass` and unmapped `.txt`)
  in_mlir = tmp_path / "prog.mlir"
  in_mlir.write_text('sw.func {sym_name = "f"} { sw.return }', encoding="utf-8")
  assert handle_convert(CliConvertOptions(input_path=in_mlir, target="jax", output_path=tmp_path / "m.py")) == 0

  in_sass = tmp_path / "kern.nvidia_sass"
  in_sass.write_text("FADD R0, R1, R2;", encoding="utf-8")
  assert handle_convert(CliConvertOptions(input_path=in_sass, target="rdna", output_path=tmp_path / "k.s")) == 0

  in_txt = tmp_path / "notes.txt"
  in_txt.write_text("x = 1", encoding="utf-8")
  assert handle_convert(CliConvertOptions(input_path=in_txt, output_path=tmp_path / "n.py")) == 0

  # Directory conversion: missing output_path error, empty dir, and populated dir with batch JSON trace
  src_dir = tmp_path / "src_dir"
  src_dir.mkdir()
  assert handle_convert(CliConvertOptions(input_path=src_dir, output_path=None)) == 1

  out_dir = tmp_path / "out_dir"
  assert handle_convert(CliConvertOptions(input_path=src_dir, output_path=out_dir)) == 0

  (src_dir / "m1.py").write_text(f"import torch{_NL}z = torch.abs(a){_NL}", encoding="utf-8")
  assert handle_convert(CliConvertOptions(input_path=src_dir, output_path=out_dir, source="torch", target="jax")) == 0
  batch_trace = tmp_path / "batch_trace.json"
  assert (
    handle_convert(
      CliConvertOptions(
        input_path=src_dir,
        output_path=out_dir,
        source="torch",
        target="jax",
        json_trace_path=batch_trace,
      )
    )
    == 0
  )
  assert (out_dir / "m1.py").exists()


def test_convert_single_file_verify_plugins_errors_and_summary(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verify `--verify` harness execution, plugin path loading, error handling, and `_print_batch_summary`.

  Args:
      tmp_path: Pytest temporary directory fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  in_py = tmp_path / "in.py"
  in_py.write_text("x = 1", encoding="utf-8")
  out_py = tmp_path / "out.py"
  sem = SemanticsManager(load_defaults=False)
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")

  # `--verify` pass (with output file and dry-run without output file) and `--verify` failure
  with (
    patch("ml_switcheroo.cli.handlers.convert.HarnessGenerator") as mock_harness,
    patch("ml_switcheroo.cli.handlers.convert.subprocess.run") as mock_run,
  ):
    mock_run.return_value.returncode = 0
    ok_verify = _convert_single_file(in_py, out_py, sem, True, cfg, None)
    assert ok_verify.success is True
    mock_harness.return_value.generate.assert_called_once()

    ok_dry_verify = _convert_single_file(in_py, None, sem, True, cfg, None)
    assert ok_dry_verify.success is True

    mock_run.return_value.returncode = 1
    fail_verify = _convert_single_file(in_py, out_py, sem, True, cfg, None)
    assert "Verification Harness Failed" in fail_verify.errors

  # Trace write failure (directory path) is non-fatal, engine failure/exception bubbles up
  bad_trace_dir = tmp_path / "trace_is_dir"
  bad_trace_dir.mkdir()
  assert _convert_single_file(in_py, out_py, sem, False, cfg, bad_trace_dir).success is True

  with patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as mock_eng_cls:
    mock_eng_cls.return_value.run.return_value = ConversionResult(code="", errors=["bad"], success=False)
    assert _convert_single_file(in_py, out_py, sem, False, cfg, None).success is False
    assert handle_convert(CliConvertOptions(input_path=in_py, output_path=out_py)) == 1

    mock_eng_cls.return_value.run.side_effect = RuntimeError("Engine Boom")
    exc_res = _convert_single_file(in_py, None, sem, False, cfg, None)
    assert exc_res.success is False
    assert "Engine Boom" in exc_res.errors[0]

  # Plugin path loading (count > 0 and count == 0) and special neither-file-nor-dir path
  cfg_with_plug = RuntimeConfig(plugin_paths=[Path("custom_plugs")])
  for loaded_cnt in (1, 0):
    with (
      patch("ml_switcheroo.cli.handlers.convert.RuntimeConfig.load", return_value=cfg_with_plug),
      patch("ml_switcheroo.cli.handlers.convert.load_plugins", return_value=loaded_cnt),
    ):
      assert handle_convert(CliConvertOptions(input_path=in_py, output_path=out_py)) == 0

  with patch("pathlib.Path.is_file", return_value=False), patch("pathlib.Path.is_dir", return_value=False):
    assert handle_convert(CliConvertOptions(input_path=in_py)) == 0

  # `_print_batch_summary`: mixed issues vs perfect batch
  _print_batch_summary(
    {
      "ok.py": ConversionResult(success=True, code=""),
      "fail.py": ConversionResult(success=False, code="", errors=["Bad"]),
      "warn.py": ConversionResult(success=True, code="", errors=["Warn"]),
    }
  )
  mixed_out = capsys.readouterr().out
  assert "1 Passed" in mixed_out
  assert "2 with Issues" in mixed_out

  _print_batch_summary({"ok.py": ConversionResult(success=True, code="")})
  assert "converted perfectly" in capsys.readouterr().out
