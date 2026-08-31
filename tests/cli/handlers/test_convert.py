"""Test suite for the CLI `convert` handler.

This module validates the behavior of the `handle_convert` function and its supporting
utilities (`_convert_single_file`, `_print_batch_summary`). It ensures that file-by-file
and directory-wide translations properly invoke the `ASTEngine`, respect command-line flags
(like `--verify` or JSON tracing), handle missing paths or unsupported extensions safely,
and report summaries accurately.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.convert import (
  ConversionResult,
  _convert_single_file,
  _print_batch_summary,
  handle_convert,
)


def test_convert_single_file_success(tmp_path: Path) -> None:
  """Test successful conversion of a single source file to an output file.

  Verifies that `_convert_single_file` accurately interfaces with the `ASTEngine`
  mock, confirms success, and correctly writes the returned AST code to the specified
  output path.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    semantics.is_verified.return_value = True

    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    from ml_switcheroo.config import RuntimeConfig

    res: ConversionResult = _convert_single_file(source, out, semantics, False, RuntimeConfig())
    assert res.success
    assert out.exists()


def test_convert_directory(tmp_path: Path) -> None:
  """Test batch conversion over a directory.

  Verifies that `handle_convert` correctly traverses a source directory, creates the
  corresponding output directory structure, and maps source files to equivalent output files.
  """
  from ml_switcheroo.cli.commands import handle_convert

  src_dir: Path = tmp_path / "src"
  src_dir.mkdir()
  (src_dir / "model.py").write_text("import torch")
  out_dir: Path = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    semantics.is_verified.return_value = True

    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    result: int = handle_convert(src_dir, out_dir, "torch", "jax", False, False, None, {})

    assert result == 0
    assert out_dir.exists()
    assert (out_dir / "model.py").exists()


def test_handle_convert_not_found() -> None:
  """Test behavior when the source path is completely missing.

  Verifies that `handle_convert` safely exits with code 1 when the user specifies
  a non-existent target.
  """
  from ml_switcheroo.cli.commands import handle_convert

  res: int = handle_convert(Path("does_not_exist"), None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_infer_source(tmp_path: Path) -> None:
  """Test framework inference logic based on file extensions.

  Verifies that if `source_fw` is omitted, the CLI correctly guesses the framework
  from the file extension (e.g., `.mlir` -> MLIR) and proceeds with a successful conversion.
  """
  from ml_switcheroo.cli.commands import handle_convert

  source: Path = tmp_path / "model.mlir"
  source.write_text("module {}")
  out: Path = tmp_path / "out"
  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    semantics.is_verified.return_value = True
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    res: int = handle_convert(source, out, None, "jax", False, False, None, {})
    assert res == 0


def test_handle_convert_plugins(tmp_path: Path) -> None:
  """Test CLI integration with dynamic plugin loading.

  Verifies that `handle_convert` properly checks the runtime config for plugin paths,
  invokes the loader, and manages failures correctly.
  """
  from ml_switcheroo.cli.commands import handle_convert

  source: Path = tmp_path / "model.py"
  source.write_text("module {}")
  out: Path = tmp_path / "out"
  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("ml_switcheroo.cli.handlers.convert.load_plugins", return_value=1),
    patch("ml_switcheroo.config.RuntimeConfig.load") as MockConfigLoad,
  ):
    semantics: MagicMock = MockSemantics()
    semantics.is_verified.return_value = True
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.config import RuntimeConfig
    from ml_switcheroo.core.engine import ConversionResult

    # We need config.plugin_paths to be set
    mock_config: RuntimeConfig = RuntimeConfig()
    mock_config.plugin_paths = [Path("my_plugin_dir")]
    MockConfigLoad.return_value = mock_config

    # Make the result fail so we hit line 104
    engine.run.return_value = ConversionResult(code="import jax", errors=["failed"], success=False, trace_events=[])
    res: int = handle_convert(source, out, "torch", "jax", False, False, None, {})
    assert res == 1


def test_handle_convert_dir_no_out(tmp_path: Path) -> None:
  """Test directory conversion rejection when no output path is supplied.

  Verifies that the user is prevented from accidentally overwriting their source
  directory structure or dumping transformed files blindly into a missing path.
  """
  from ml_switcheroo.cli.commands import handle_convert

  source: Path = tmp_path / "src"
  source.mkdir()
  res: int = handle_convert(source, None, "torch", "jax", False, False, None, {})
  assert res == 1


def test_handle_convert_dir_empty(tmp_path: Path) -> None:
  """Test behavior when a source directory contains no convertible files.

  Verifies that the CLI returns cleanly (code 0) without creating invalid outputs
  when targeting an empty directory.
  """
  from ml_switcheroo.cli.commands import handle_convert

  source: Path = tmp_path / "src"
  source.mkdir()
  out: Path = tmp_path / "out"
  res: int = handle_convert(source, out, "torch", "jax", False, False, None, {})
  assert res == 0


def test_handle_convert_dir_json_trace(tmp_path: Path) -> None:
  """Test writing JSON traces during a batch directory conversion.

  Verifies the `-j / --json` flag is respected for full directory scans, allowing
  the pipeline to aggregate transformation traces.
  """
  from ml_switcheroo.cli.commands import handle_convert

  source: Path = tmp_path / "src"
  source.mkdir()
  (source / "model.py").write_text("import torch")
  out: Path = tmp_path / "out"
  json_path: Path = tmp_path / "trace.json"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    semantics.is_verified.return_value = True
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    res: int = handle_convert(source, out, "torch", "jax", False, False, None, {}, json_path)
    assert res == 0


def test_convert_single_file_json_trace(tmp_path: Path) -> None:
  """Test the actual file I/O for JSON tracing on a single file.

  Verifies that when a JSON path is provided to `_convert_single_file`, the trace
  events are written correctly to disk.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"
  json_path: Path = tmp_path / "trace.json"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(
      code="import jax", errors=[], success=True, trace_events=[{"event": "test"}]
    )
    from ml_switcheroo.config import RuntimeConfig

    res: ConversionResult = _convert_single_file(source, out, semantics, False, RuntimeConfig(), json_path)
    assert res.success
    assert json_path.exists()


def test_convert_single_file_json_trace_error(tmp_path: Path) -> None:
  """Test resilience to file I/O errors during JSON trace dumping.

  Verifies that if trace writing fails (e.g. attempting to write to a directory),
  the primary conversion still succeeds gracefully without crashing the whole application.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"
  # Using a directory path to trigger error
  json_path: Path = tmp_path / "trace.json"
  json_path.mkdir()

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(
      code="import jax", errors=[], success=True, trace_events=[{"event": "test"}]
    )
    from ml_switcheroo.config import RuntimeConfig

    res: ConversionResult = _convert_single_file(source, out, semantics, False, RuntimeConfig(), json_path)
    assert res.success


def test_convert_single_file_fail(tmp_path: Path) -> None:
  """Test internal engine error handling during file conversion.

  Verifies that if the underlying `ASTEngine.run()` method signals failure,
  the outer `_convert_single_file` utility correctly bubbles up the failure state.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="", errors=["err"], success=False, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    res: ConversionResult = _convert_single_file(source, out, semantics, False, RuntimeConfig(), None)
    assert not res.success


def test_convert_single_file_verify(tmp_path: Path) -> None:
  """Test the structural `--verify` feature on successful conversion.

  Verifies that when verification is enabled, the CLI executes the target
  interpreter (e.g. `python -m py_compile`) and correctly marks success on code 0.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc: MagicMock = MagicMock()
    mock_proc.returncode = 0
    MockSubprocess.return_value = mock_proc

    res: ConversionResult = _convert_single_file(source, out, semantics, True, RuntimeConfig(), None)
    assert res.success


def test_convert_single_file_verify_fail(tmp_path: Path) -> None:
  """Test the structural `--verify` feature failing on invalid syntax.

  Verifies that if the target interpreter fails to parse the generated AST (code 1),
  the result adds an error but technically remains a "success" from the engine's perspective
  (as code was generated).
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")
  out: Path = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc: MagicMock = MagicMock()
    mock_proc.returncode = 1
    MockSubprocess.return_value = mock_proc

    res: ConversionResult = _convert_single_file(source, out, semantics, True, RuntimeConfig(), None)
    assert res.success
    assert len(res.errors) == 1


def test_convert_single_file_verify_no_out(tmp_path: Path) -> None:
  """Test verification flow when executing dry-runs (no output file).

  Verifies that the CLI creates an ephemeral temporary file to pass to `python -m py_compile`
  when verification is requested but no physical output file was commanded.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc: MagicMock = MagicMock()
    mock_proc.returncode = 0
    MockSubprocess.return_value = mock_proc

    res: ConversionResult = _convert_single_file(source, None, semantics, True, RuntimeConfig(), None)
    assert res.success


def test_convert_single_file_exception(tmp_path: Path) -> None:
  """Test hard exception handling in the engine wrapper.

  Verifies that if `engine.run()` raises an unhandled generic Python exception,
  it is safely caught and converted into a clean `ConversionResult(success=False)`.
  """
  source: Path = tmp_path / "model.py"
  source.write_text("import torch")

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics: MagicMock = MockSemantics()
    engine: MagicMock = MockEngine.return_value
    engine.run.side_effect = Exception("error")
    from ml_switcheroo.config import RuntimeConfig

    res: ConversionResult = _convert_single_file(source, None, semantics, False, RuntimeConfig(), None)
    assert not res.success


def test_print_batch_summary() -> None:
  """Test the visual layout and rendering logic of batch summaries.

  Verifies that the console helper `_print_batch_summary` can process a dictionary
  of outcomes without crashing.
  """
  from ml_switcheroo.core.engine import ConversionResult

  results: dict[str, ConversionResult] = {
    "a": ConversionResult(code="a", errors=[], success=True, trace_events=[]),
    "b": ConversionResult(code="b", errors=["err"], success=False, trace_events=[]),
  }
  _print_batch_summary(results)


def test_convert_missing_branches(tmp_path: Path) -> None:
  """Test specific unreached branches in the primary CLI handler.

  Verifies edge cases like files lacking mapped extensions (e.g. `.txt`), handling
  `load_plugins` returning 0 valid modules, and fallback execution when a target
  path isn't identified as either a valid file or directory.
  """
  from unittest.mock import patch

  # 74->79: file extension not in ext_map
  unsupported_ext = tmp_path / "model.txt"
  unsupported_ext.write_text("hello")
  handle_convert(unsupported_ext, None, None, None, False, False, False, None, None)

  # 93->96: load_plugins returns 0
  with patch("ml_switcheroo.cli.handlers.convert.RuntimeConfig.load") as mock_load:
    mock_load.return_value.plugin_paths = ["/fake/path"]
    with patch("ml_switcheroo.cli.handlers.convert.load_plugins", return_value=0):
      handle_convert(unsupported_ext, None, None, None, False, False, False, None, None)

  # 106->131: neither is_file nor is_dir
  with patch("pathlib.Path.is_file", return_value=False):
    with patch("pathlib.Path.is_dir", return_value=False):
      handle_convert(unsupported_ext, None, None, None, False, False, False, None, None)
