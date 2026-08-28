"""Test module."""

import libcst as cst
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig


def test_engine_init() -> None:
  """Test element."""
  engine = ASTEngine(source="torch", target="jax", strict_mode=True, intermediate="mlir")
  assert engine.source == "torch"
  assert engine.target == "jax"
  assert engine.strict_mode is True
  assert engine.config.intermediate == "mlir"


def test_engine_init_with_config() -> None:
  """Test element."""
  config = RuntimeConfig(source_framework="keras", target_framework="mlx")
  engine = ASTEngine(config=config, intermediate="tikz")
  assert engine.source == "keras"
  assert engine.target == "mlx"
  assert engine.config.intermediate == "tikz"


@patch("ml_switcheroo.core.engine.SemanticsManager.load_validation_report")
def test_engine_init_validation_report(mock_load: MagicMock) -> None:
  """Test element."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report="report.json")
  ASTEngine(config=config)
  from pathlib import Path

  mock_load.assert_called_once_with(Path("report.json"))


def test_engine_parse_to_source() -> None:
  """Test element."""
  engine = ASTEngine()
  code: str = "x = 1"
  tree: cst.Module = engine.parse(code)
  assert isinstance(tree, cst.Module)
  assert engine.to_source(tree) == "x = 1"  # to_source without newline


@patch("ml_switcheroo.core.engine.MermaidGenerator.generate")
def test_engine_graph_to_mermaid(mock_gen: MagicMock) -> None:
  """Test element."""
  mock_gen.return_value = "graph TD;"
  engine = ASTEngine()
  tree: cst.Module = engine.parse("x=1")
  assert engine._graph_to_mermaid(tree) == "graph TD;"


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_engine_run_isa(mock_run_compiler: MagicMock) -> None:
  """Test element."""
  mock_run_compiler.return_value = ConversionResult(success=True)
  engine = ASTEngine(source="sass", target="html")
  res: ConversionResult = engine.run("code")
  mock_run_compiler.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_engine_run_sharding(mock_run_compiler: MagicMock) -> None:
  """Test element."""
  mock_run_compiler.return_value = ConversionResult(success=True)
  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
  engine = ASTEngine(config=config)
  res: ConversionResult = engine.run("code")
  mock_run_compiler.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_engine_run_stablehlo(mock_ingest: MagicMock, mock_emitter_class: MagicMock) -> None:
  """Test element."""
  mock_ingest.return_value = cst.parse_module("pass")
  mock_emitter = mock_emitter_class.return_value
  mock_emitter.convert.return_value.to_text.return_value = "mlir_code"

  engine = ASTEngine(source="torch", target="stablehlo")
  res: ConversionResult = engine.run("code")
  assert res.code == "mlir_code"
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_engine_run_rewriter(mock_run_rewriter: MagicMock) -> None:
  """Test element."""
  mock_run_rewriter.return_value = ConversionResult(success=True)
  engine = ASTEngine(source="torch", target="jax")
  res: ConversionResult = engine.run("code")
  mock_run_rewriter.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_engine_run_exception(mock_run_rewriter: MagicMock) -> None:
  """Test element."""
  mock_run_rewriter.side_effect = ValueError("Boom")
  engine = ASTEngine(source="torch", target="jax")
  res: ConversionResult = engine.run("code")
  assert res.success is False
  assert res.errors is not None
  assert "Boom" in res.errors[0]
