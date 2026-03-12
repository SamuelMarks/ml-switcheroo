import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.import_fixer import ResolutionPlan
import libcst as cst
from pathlib import Path


def test_engine_init_coverage():
  cfg = RuntimeConfig(strict_mode=False)
  engine = ASTEngine(config=cfg, intermediate="onnx")
  assert engine.config.intermediate == "onnx"

  cfg2 = RuntimeConfig(strict_mode=False)
  cfg2.validation_report = Path("/tmp/report.json")
  with patch.object(SemanticsManager, "load_validation_report") as mock_load:
    engine2 = ASTEngine(config=cfg2)
    mock_load.assert_called_once_with(Path("/tmp/report.json"))


def test_engine_run_exception_coverage():
  engine = ASTEngine(source="torch", target="jax")
  with patch.object(engine, "_run_rewriter_pipeline", side_effect=Exception("mocked error")):
    res = engine.run("def foo(): pass")
    assert res.success is False
    assert "mocked error" in res.errors[0]


def test_engine_parse_coverage():
  engine = ASTEngine()
  tree = engine.parse("x = 1")
  assert isinstance(tree, cst.Module)


class MockBackend:
  def __init__(self, *args, **kwargs):
    pass

  def compile(self, graph):
    return "compiled code"


def get_tracer_mock():
  m = MagicMock()
  m.export.return_value = []
  return m


def test_compiler_pipeline_coverage():
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", side_effect=Exception("mocked err")),
    patch("ml_switcheroo.core.engine.PythonFrontend") as mock_frontend,
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_frontend.return_value.parse_to_graph.return_value = MagicMock()
    mock_get_backend.return_value = MockBackend
    res = engine._run_compiler_pipeline("code", get_tracer_mock())
    assert res.code == "compiled code"

  engine_sass = ASTEngine(source="sass", target="rdna")
  with (
    patch("ml_switcheroo.core.engine.SassParser"),
    patch("ml_switcheroo.core.engine.SassLifter"),
    patch("ml_switcheroo.core.engine.get_backend_class", return_value=MockBackend),
  ):
    engine_sass._run_compiler_pipeline("code", get_tracer_mock())

  engine_unknown = ASTEngine(source="torch", target="rdna")
  engine_unknown.source = "unknown_isa"
  with patch("ml_switcheroo.core.engine.is_isa_source", return_value=True):
    with pytest.raises(NotImplementedError):
      engine_unknown._run_compiler_pipeline("code", get_tracer_mock())

  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine_shard = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code"),
    patch("ml_switcheroo.core.engine.PythonFrontend"),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.engine.get_backend_class", return_value=MockBackend),
  ):
    engine_shard._run_compiler_pipeline("code", get_tracer_mock())

  engine_shard_torch = ASTEngine(config=cfg, source="jax", target="torch")
  with (
    patch("ml_switcheroo.core.engine.ingest_code"),
    patch("ml_switcheroo.core.engine.PythonFrontend"),
    patch("ml_switcheroo.core.engine.get_backend_class", return_value=MockBackend),
  ):
    engine_shard_torch._run_compiler_pipeline("code", get_tracer_mock())


class MockPythonBackend:
  def __init__(self, *args, **kwargs):
    pass

  def compile(self, graph):
    return "compiled py code"


def test_compiler_pipeline_backend_coverage():
  engine = ASTEngine(source="torch", target="jax")
  with (
    patch("ml_switcheroo.core.engine.ingest_code"),
    patch("ml_switcheroo.core.engine.PythonFrontend"),
    patch("ml_switcheroo.core.engine.get_backend_class", return_value=None),
  ):
    with pytest.raises(ValueError, match="No backend found for jax"):
      engine._run_compiler_pipeline("code", get_tracer_mock())

  engine_py = ASTEngine(source="torch", target="jax")
  with (
    patch("ml_switcheroo.core.engine.ingest_code"),
    patch("ml_switcheroo.core.engine.PythonFrontend"),
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_py = MockPythonBackend
    mock_py.__name__ = "PythonBackend"
    mock_get_backend.return_value = mock_py
    res = engine_py._run_compiler_pipeline("code", get_tracer_mock())
    assert res.code == "compiled py code"


def test_rewriter_pipeline_coverage():
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as mock_patcher,
    patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_extractor.return_value.graph.nodes = [1]
    mock_differ.return_value.diff.return_value = True

    engine._run_rewriter_pipeline("code", get_tracer_mock())

  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine_shard = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
  ):
    mock_extractor.return_value.graph.nodes = [1]
    mock_differ.return_value.diff.return_value = False
    engine_shard._run_rewriter_pipeline("code", get_tracer_mock())

  engine_err = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor", side_effect=Exception("extractor error")),
  ):
    tracer = get_tracer_mock()
    engine_err._run_rewriter_pipeline("code", tracer)
    tracer.log_warning.assert_called()

  cfg_import = RuntimeConfig(strict_mode=False)
  cfg_import.enable_import_fixer = True
  engine_import = ASTEngine(config=cfg_import, source="torch", target="jax")
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.UsageScanner.get_result", return_value=False),
    patch(
      "ml_switcheroo.core.engine.ImportResolver.resolve",
      return_value=ResolutionPlan(
        path_to_alias={},
        required_imports=set(),
      ),
    ),
  ):
    engine_import._run_rewriter_pipeline("code", get_tracer_mock())

  engine_mlir = ASTEngine(source="torch", target="mlir")
  with patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")):
    tracer = get_tracer_mock()
    engine_mlir._run_rewriter_pipeline("code", tracer)

  engine_hatch = ASTEngine(source="torch", target="jax")
  with patch(
    "ml_switcheroo.core.engine.ingest_code",
    return_value=cst.parse_module("# <SWITCHEROO_FAILED_TO_TRANS>\ndef foo(): pass"),
  ):
    res = engine_hatch._run_rewriter_pipeline("code", get_tracer_mock())
    assert len(res.errors) > 0
    assert "Escape Hatches Detected" in res.errors[0]

  engine_strict = ASTEngine(source="torch", target="jax", strict_mode=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.StructuralLinter") as mock_linter,
  ):
    mock_linter.return_value.check.return_value = ["linter err"]
    res = engine_strict._run_rewriter_pipeline("code", get_tracer_mock())
    assert "linter err" in res.errors


def test_rewriter_sharding_torch_target():
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
  ):
    mock_extractor.return_value.graph.nodes = [1]
    mock_differ.return_value.diff.return_value = False
    engine._run_rewriter_pipeline("code", get_tracer_mock())
