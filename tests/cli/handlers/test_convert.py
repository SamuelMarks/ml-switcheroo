"""Docstring."""

from unittest.mock import MagicMock, patch
from pathlib import Path
from ml_switcheroo.cli.handlers.convert import _convert_single_file, _print_batch_summary


def test_convert_single_file_success(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    semantics.is_verified.return_value = True

    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    from ml_switcheroo.config import RuntimeConfig

    res = _convert_single_file(source, out, semantics, False, RuntimeConfig())
    assert res.success
    assert out.exists()


def test_convert_directory(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  src_dir = tmp_path / "src"
  src_dir.mkdir()
  (src_dir / "model.py").write_text("import torch")
  out_dir = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    semantics.is_verified.return_value = True

    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    result = handle_convert(src_dir, out_dir, "torch", "jax", False, False, None, {})

    assert result == 0
    assert out_dir.exists()
    assert (out_dir / "model.py").exists()


def test_handle_convert_not_found():
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  res = handle_convert(Path("does_not_exist"), None, None, None, False, False, None, {})
  assert res == 1


def test_handle_convert_infer_source(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  source = tmp_path / "model.mlir"
  source.write_text("module {}")
  out = tmp_path / "out"
  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    semantics.is_verified.return_value = True
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    res = handle_convert(source, out, None, "jax", False, False, None, {})
    assert res == 0


def test_handle_convert_plugins(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  source = tmp_path / "model.py"
  source.write_text("module {}")
  out = tmp_path / "out"
  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("ml_switcheroo.cli.handlers.convert.load_plugins", return_value=1),
    patch("ml_switcheroo.config.RuntimeConfig.load") as MockConfigLoad,
  ):
    semantics = MockSemantics()
    semantics.is_verified.return_value = True
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult
    from ml_switcheroo.config import RuntimeConfig

    # We need config.plugin_paths to be set
    mock_config = RuntimeConfig()
    mock_config.plugin_paths = [Path("my_plugin_dir")]
    MockConfigLoad.return_value = mock_config

    # Make the result fail so we hit line 104
    engine.run.return_value = ConversionResult(code="import jax", errors=["failed"], success=False, trace_events=[])
    res = handle_convert(source, out, "torch", "jax", False, False, None, {})
    assert res == 1


def test_handle_convert_dir_no_out(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  source = tmp_path / "src"
  source.mkdir()
  res = handle_convert(source, None, "torch", "jax", False, False, None, {})
  assert res == 1


def test_handle_convert_dir_empty(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  source = tmp_path / "src"
  source.mkdir()
  out = tmp_path / "out"
  res = handle_convert(source, out, "torch", "jax", False, False, None, {})
  assert res == 0


def test_handle_convert_dir_json_trace(tmp_path):
  """Docstring."""
  from ml_switcheroo.cli.commands import handle_convert

  source = tmp_path / "src"
  source.mkdir()
  (source / "model.py").write_text("import torch")
  out = tmp_path / "out"
  json_path = tmp_path / "trace.json"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    semantics.is_verified.return_value = True
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])

    res = handle_convert(source, out, "torch", "jax", False, False, None, {}, json_path)
    assert res == 0


def test_convert_single_file_json_trace(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"
  json_path = tmp_path / "trace.json"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(
      code="import jax", errors=[], success=True, trace_events=[{"event": "test"}]
    )
    from ml_switcheroo.config import RuntimeConfig

    res = _convert_single_file(source, out, semantics, False, RuntimeConfig(), json_path)
    assert res.success
    assert json_path.exists()


def test_convert_single_file_json_trace_error(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"
  # Using a directory path to trigger error
  json_path = tmp_path / "trace.json"
  json_path.mkdir()

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(
      code="import jax", errors=[], success=True, trace_events=[{"event": "test"}]
    )
    from ml_switcheroo.config import RuntimeConfig

    res = _convert_single_file(source, out, semantics, False, RuntimeConfig(), json_path)
    assert res.success


def test_convert_single_file_fail(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="", errors=["err"], success=False, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    res = _convert_single_file(source, out, semantics, False, RuntimeConfig(), None)
    assert not res.success


def test_convert_single_file_verify(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    MockSubprocess.return_value = mock_proc

    res = _convert_single_file(source, out, semantics, True, RuntimeConfig(), None)
    assert res.success


def test_convert_single_file_verify_fail(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")
  out = tmp_path / "out"

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    MockSubprocess.return_value = mock_proc

    res = _convert_single_file(source, out, semantics, True, RuntimeConfig(), None)
    assert res.success
    assert len(res.errors) == 1


def test_convert_single_file_verify_no_out(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
    patch("subprocess.run") as MockSubprocess,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    from ml_switcheroo.core.engine import ConversionResult

    engine.run.return_value = ConversionResult(code="import jax", errors=[], success=True, trace_events=[])
    from ml_switcheroo.config import RuntimeConfig

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    MockSubprocess.return_value = mock_proc

    res = _convert_single_file(source, None, semantics, True, RuntimeConfig(), None)
    assert res.success


def test_convert_single_file_exception(tmp_path):
  """Docstring."""
  source = tmp_path / "model.py"
  source.write_text("import torch")

  with (
    patch("ml_switcheroo.cli.handlers.convert.SemanticsManager") as MockSemantics,
    patch("ml_switcheroo.cli.handlers.convert.ASTEngine") as MockEngine,
  ):
    semantics = MockSemantics()
    engine = MockEngine.return_value
    engine.run.side_effect = Exception("error")
    from ml_switcheroo.config import RuntimeConfig

    res = _convert_single_file(source, None, semantics, False, RuntimeConfig(), None)
    assert not res.success


def test_print_batch_summary():
  """Docstring."""
  from ml_switcheroo.core.engine import ConversionResult

  results = {
    "a": ConversionResult(code="a", errors=[], success=True, trace_events=[]),
    "b": ConversionResult(code="b", errors=["err"], success=False, trace_events=[]),
  }
  _print_batch_summary(results)
