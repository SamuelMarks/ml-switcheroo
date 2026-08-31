"""Test module."""

import typing
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.engine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.import_fixer import ResolutionPlan
from ml_switcheroo.semantics.manager import SemanticsManager


def test_engine_init() -> None:
  """Docstring."""
  engine = ASTEngine(source="torch", target="jax", strict_mode=True, intermediate="mlir")
  assert engine.source == "torch"
  assert engine.target == "jax"
  assert engine.strict_mode is True
  assert engine.config.intermediate == "mlir"


def test_engine_init_with_config() -> None:
  """Docstring."""
  config = RuntimeConfig(source_framework="keras", target_framework="mlx")
  engine = ASTEngine(config=config, intermediate="tikz")
  assert engine.source == "keras"
  assert engine.target == "mlx"
  assert engine.config.intermediate == "tikz"


@patch("ml_switcheroo.core.engine.SemanticsManager.load_validation_report")
def test_engine_init_validation_report(mock_load: MagicMock) -> None:
  """Docstring."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report="report.json")
  ASTEngine(config=config)
  from pathlib import Path

  mock_load.assert_called_once_with(Path("report.json"))


def test_engine_parse_to_source() -> None:
  """Docstring."""
  engine = ASTEngine()
  code: str = "x = 1"
  tree: cst.Module = engine.parse(code)
  assert isinstance(tree, cst.Module)
  assert engine.to_source(tree) == "x = 1"  # to_source without newline


@patch("ml_switcheroo.core.engine.MermaidGenerator.generate")
def test_engine_graph_to_mermaid(mock_gen: MagicMock) -> None:
  """Docstring."""
  mock_gen.return_value = "graph TD;"
  engine = ASTEngine()
  tree: cst.Module = engine.parse("x=1")
  assert engine._graph_to_mermaid(tree) == "graph TD;"


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_engine_run_isa(mock_run_compiler: MagicMock) -> None:
  """Docstring."""
  mock_run_compiler.return_value = ConversionResult(success=True)
  engine = ASTEngine(source="sass", target="html")
  res: ConversionResult = engine.run("code")
  mock_run_compiler.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_engine_run_sharding(mock_run_compiler: MagicMock) -> None:
  """Docstring."""
  mock_run_compiler.return_value = ConversionResult(success=True)
  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
  engine = ASTEngine(config=config)
  res: ConversionResult = engine.run("code")
  mock_run_compiler.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_engine_run_stablehlo(mock_ingest: MagicMock, mock_emitter_class: MagicMock) -> None:
  """Docstring."""
  mock_ingest.return_value = cst.parse_module("pass")
  mock_emitter = mock_emitter_class.return_value
  mock_emitter.convert.return_value.to_text.return_value = "mlir_code"

  engine = ASTEngine(source="torch", target="stablehlo")
  res: ConversionResult = engine.run("code")
  assert res.code == "mlir_code"
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_engine_run_rewriter(mock_run_rewriter: MagicMock) -> None:
  """Docstring."""
  mock_run_rewriter.return_value = ConversionResult(success=True)
  engine = ASTEngine(source="torch", target="jax")
  res: ConversionResult = engine.run("code")
  mock_run_rewriter.assert_called_once()
  assert res.success is True


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_engine_run_exception(mock_run_rewriter: MagicMock) -> None:
  """Docstring."""
  mock_run_rewriter.side_effect = ValueError("Boom")
  engine = ASTEngine(source="torch", target="jax")
  res: ConversionResult = engine.run("code")
  assert res.success is False
  assert res.errors is not None
  assert "Boom" in res.errors[0]


# --- Merged from test_engine_missing.py ---


def test_astengine_init_missing_branches() -> None:
  """Docstring."""
  engine1 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"

  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2 = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  ASTEngine(config=cfg2)
  assert not cfg2.validation_report


def test_astengine_stablehlo_missing_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.core.engine

  engine = ASTEngine(source="torch", target="stablehlo")
  import libcst as cst

  monkeypatch.setattr(ml_switcheroo.core.engine, "ingest_code", lambda *args: cst.parse_module("def foo(): pass"))

  class MockEmitter:
    """Docstring."""

    def __init__(self, semantics: typing.Any) -> None:
      """Docstring."""
      pass

    def convert(self, tree: typing.Any) -> typing.Any:
      """Docstring."""

      class TextObj:
        """Docstring."""

        def to_text(self) -> str:
          """Docstring."""
          return "mlir"

      return TextObj()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  monkeypatch.setattr("ml_switcheroo.core.engine.get_adapter", lambda *args: None)
  result: ConversionResult = engine.run("def f(): pass")
  assert result.success


def test_astengine_sass_unsupported_missing_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.core.engine

  # To bypass pydantic validation, we just create the engine normally and override self.target
  engine = ASTEngine(source="sass", target="jax")
  engine.target = "unsupported"

  # We must mock get_backend_class so it hits the check and raises ValueError? Wait, if we mock it to return None
  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)

  with (
    mock.patch("ml_switcheroo.core.compiler.frontends.sass.SassParser") as mock_parser,
    mock.patch("ml_switcheroo.core.compiler.frontends.sass.SassLifter") as mock_lifter,
  ):
    mock_parser.return_value.parse.return_value.statements = []
    mock_lifter.return_value.lift.return_value = mock.MagicMock()
    result: ConversionResult = engine.run("sass code")
    assert not result.success
    assert result.errors is not None
    assert any("No backend found" in e for e in result.errors)


# --- Merged from test_engine_gap19.py ---


def test_astengine_stablehlo_branch(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  # Test the branch elif self.target == "stablehlo" without mocking ingest_code
  engine = ASTEngine(source="torch", target="stablehlo")
  # Provide simple valid python code
  code: str = "def f(): pass"

  # We mock StableHloEmitter so we don't need its implementation
  class MockEmitter:
    """Docstring."""

    def __init__(self, semantics: typing.Any) -> None:
      """Docstring."""
      pass

    def convert(self, tree: typing.Any) -> typing.Any:
      """Docstring."""

      class TextObj:
        """Docstring."""

        def to_text(self) -> str:
          """Docstring."""
          return "mlir"

      return TextObj()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  # also we need to avoid the 'self.target' check in some places maybe?
  result: ConversionResult = engine.run(code)
  assert result.success
  assert result.code == "mlir"


# --- Merged from test_engine_gap4.py ---


def test_rewriter_loopback_sharding_actual() -> None:
  """Verifies the behavior of rewriter loopback sharding actual."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  cfg.enable_graph_optimization = True
  engine = ASTEngine(config=cfg, source="torch", target="jax")
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}
    mock_differ.return_value.diff.return_value = [1]
    with (
      patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()
  cfg.enable_graph_optimization = True
  engine_torch = ASTEngine(config=cfg, source="jax", target="torch")
  engine_torch.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}
    mock_differ.return_value.diff.return_value = [1]
    with (
      patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine_torch._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()


# --- Merged from test_engine_gap.py ---


def test_engine_init_coverage() -> None:
  """Verifies the behavior of engine initialization coverage."""
  cfg = RuntimeConfig(strict_mode=False)
  engine = ASTEngine(config=cfg, intermediate="onnx")
  assert engine.config.intermediate == "onnx"
  cfg2 = RuntimeConfig(strict_mode=False)
  cfg2.validation_report = str(Path("/tmp/report.json"))
  with patch.object(SemanticsManager, "load_validation_report") as mock_load:
    ASTEngine(config=cfg2)
    mock_load.assert_called_once_with("/tmp/report.json")


def test_engine_run_exception_coverage() -> None:
  """Verifies the behavior of engine run correctly handling an exception coverage."""
  engine = ASTEngine(source="torch", target="jax")
  with patch.object(engine, "_run_rewriter_pipeline", side_effect=Exception("mocked error")):
    res: ConversionResult = engine.run("def foo(): pass")
    assert res.success is False
    assert res.errors is not None
    assert "mocked error" in res.errors[0]


def test_engine_parse_coverage() -> None:
  """Verifies the behavior of engine parse coverage."""
  engine = ASTEngine()
  tree: cst.Module = engine.parse("x = 1")
  assert isinstance(tree, cst.Module)


class MockBackend:
  """Docstring."""

  def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
    """Initializes the MockBackend instance."""
    pass

  def compile(self, graph: typing.Any) -> str:
    """Mock implementation of compile."""
    return "compiled code"


def get_tracer_mock() -> MagicMock:
  """Gets tracer mock."""
  m = MagicMock()
  m.export.return_value = []
  return m


def test_compiler_pipeline_coverage() -> None:
  """Verifies the behavior of compiler pipeline coverage."""
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", side_effect=Exception("mocked err")),
    patch("ml_switcheroo.core.engine.PythonFrontend") as mock_frontend,
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_frontend.return_value.parse_to_graph.return_value = MagicMock()
    mock_get_backend.return_value = MockBackend
    res: ConversionResult = engine._run_compiler_pipeline("code", get_tracer_mock())
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
  """Docstring."""

  def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
    """Initializes the MockPythonBackend instance."""
    pass

  def compile(self, graph: typing.Any) -> str:
    """Mock implementation of compile."""
    return "compiled py code"


def test_compiler_pipeline_backend_coverage() -> None:
  """Verifies the behavior of compiler pipeline backend coverage."""
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
    res: ConversionResult = engine_py._run_compiler_pipeline("code", get_tracer_mock())
    assert res.code == "compiled py code"


def test_rewriter_pipeline_coverage() -> None:
  """Verifies the behavior of rewriter pipeline coverage."""
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_extractor.return_value.graph.nodes = [1]  # type: ignore
    mock_differ.return_value.diff.return_value = True
    engine._run_rewriter_pipeline("code", get_tracer_mock())
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine_shard = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
  ):
    mock_extractor.return_value.graph.nodes = [1]  # type: ignore
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

  class MockUsageScanner(cst.CSTVisitor):
    """Docstring."""

    def __init__(self, *args, **kwargs):
      """Docstring."""
      self.used_names = set()

  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GlobalUsageScanner", return_value=MockUsageScanner()),
    patch(
      "ml_switcheroo.core.engine.ImportResolver.resolve",
      return_value=ResolutionPlan(path_to_alias={}, required_imports=[]),
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
    res: ConversionResult = engine_hatch._run_rewriter_pipeline("code", get_tracer_mock())
    assert res.errors is not None
    assert len(res.errors) > 0
    assert "Escape Hatches Detected" in res.errors[0]
  engine_strict = ASTEngine(source="torch", target="jax", strict_mode=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.StructuralLinter") as mock_linter,
  ):
    mock_linter.return_value.check.return_value = ["linter err"]
    res = engine_strict._run_rewriter_pipeline("code", get_tracer_mock())
    assert res.errors is not None
    assert "linter err" in res.errors


def test_rewriter_sharding_torch_target() -> None:
  """Verifies the behavior of rewriter sharding PyTorch target."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
  ):
    mock_extractor.return_value.graph.nodes = [1]  # type: ignore
    mock_differ.return_value.diff.return_value = False
    engine._run_rewriter_pipeline("code", get_tracer_mock())


# --- Merged from test_engine_gap5.py ---


def test_engine_target_torch_sharding_compiler() -> None:
  """Verifies the behavior of engine target PyTorch sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          engine._run_compiler_pipeline(code, MagicMock())
          MockSharding.assert_called()


def test_engine_target_flax_sharding_compiler() -> None:
  """Verifies the behavior of engine target Flax sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="flax", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          engine._run_compiler_pipeline(code, MagicMock())
          MockSharding.assert_called()


def test_engine_target_torch_sharding_rewriter_2() -> None:
  """Verifies the behavior of engine target PyTorch sharding rewriter 2."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=None):
        with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
          MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
          with patch("ml_switcheroo.core.graph.GraphExtractor") as MockExtCls:
            MockExt = MockExtCls.return_value
            g = MagicMock()
            g.nodes = ["n1"]
            MockExt.graph = g
            MockExt.node_map = {}
            with patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"):
              with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"):
                engine._run_rewriter_pipeline(code, MagicMock())
                MockSharding.assert_called()


# --- Merged from test_engine_gap8.py ---


def test_engine_target_keras_sharding_compiler() -> None:
  """Verifies the behavior of engine target Keras sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="keras", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock()
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


def test_engine_target_keras_sharding_rewriter() -> None:
  """Verifies the behavior of engine target Keras sharding rewriter."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="keras", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=None):
        with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
          MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
          with patch("ml_switcheroo.core.graph.GraphExtractor") as MockExtCls:
            MockExt = MockExtCls.return_value
            g = MagicMock()
            g.nodes = ["n1"]
            MockExt.graph = g
            MockExt.node_map = {}
            engine._run_rewriter_pipeline(code, MagicMock())
            MockSharding.assert_called()


# --- Merged from test_engine_gap10.py ---


def test_engine_strict_mode_linter_errors() -> None:
  """Verifies the behavior of engine strict mode linter errors."""
  config = RuntimeConfig(strict_mode=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.engine.StructuralLinter.check") as MockCheck:
    MockCheck.return_value = ["linter error"]
    with patch("ml_switcheroo.core.engine.ingest_code") as MockIngest:
      mock_tree = MagicMock()
      mock_tree.code = "import jax"
      with patch("ml_switcheroo.core.engine.RewriterPipeline") as MockPipeCls:
        MockPipeCls.return_value.run.return_value = mock_tree
        with patch("ml_switcheroo.core.engine.ImportFixer"):
          mock_fixer_visit = MagicMock()
          mock_fixer_visit.code = "import jax"
          mock_tree.visit.return_value = mock_fixer_visit
          MockIngest.return_value = mock_tree
          engine.config.enable_graph_optimization = False
          res: ConversionResult = engine._run_rewriter_pipeline(code, MagicMock())
          assert res.errors is not None
          assert "linter error" in res.errors[0]


def test_engine_escape_hatches_detected() -> None:
  """Verifies the behavior of engine escape hatches detected."""
  config = RuntimeConfig()
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  with patch("ml_switcheroo.core.engine.ingest_code") as MockIngest:
    mock_tree = MagicMock()
    mock_tree.code = f"{EscapeHatch.START_MARKER} some code"
    with patch("ml_switcheroo.core.engine.RewriterPipeline") as MockPipeCls:
      MockPipeCls.return_value.run.return_value = mock_tree
      with patch("ml_switcheroo.core.engine.ImportFixer"):
        mock_fixer_visit = MagicMock()
        mock_fixer_visit.code = mock_tree.code
        mock_tree.visit.return_value = mock_fixer_visit
        MockIngest.return_value = mock_tree
        engine.config.enable_graph_optimization = False
        res: ConversionResult = engine._run_rewriter_pipeline(code, MagicMock())
        assert res.errors is not None
        assert "Escape Hatches Detected" in res.errors[0]


def test_engine_target_torch_sharding_compiler_extra() -> None:
  """Verifies the behavior of engine target PyTorch sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock(nodes=["n1"])
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


# --- Merged from test_engine_gap20.py ---


def test_run_compiler_pipeline_rdna_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.core.engine

  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)

  engine = ASTEngine(source="rdna", target="jax")
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)
  result: ConversionResult = engine.run("v_add_f32 v0, v1, v2")
  assert not result.success


def test_astengine_fusion_target_branches_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.core.engine

  class FakeBackend:
    """Docstring."""

    def __init__(self, semantics: typing.Any) -> None:
      """Docstring."""
      pass

    def set_mode(self, *args: typing.Any, **kwargs: typing.Any) -> None:
      """Docstring."""
      pass

    def compile(self, graph: typing.Any) -> typing.Any:
      """Docstring."""
      from ml_switcheroo.core.compiler.backends.base import BackendResult

      return BackendResult(code="COMP", imports=["A"], attrs=[])

  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: FakeBackend)

  cfg = RuntimeConfig(source_framework="sass", target_framework="jax", enable_graph_optimization=True)
  engine = ASTEngine(config=cfg)

  engine.target = "flax"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "flax_nnx"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "paxml"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "torch"  # hit else branch
  engine.run("v_add_f32 v0, v1, v2")


def test_astengine_sass_unsupported_target(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)
  engine = ASTEngine(source="sass", target="jax")
  result: ConversionResult = engine.run("v_add_f32 v0, v1, v2")
  assert not result.success


def test_astengine_unsupported_isa_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  engine = ASTEngine(source="torch", target="jax")
  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)
  result: ConversionResult = engine.run("def a(): pass")
  assert not result.success


def test_astengine_init_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  engine1 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"

  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2 = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  ASTEngine(config=cfg2)
  assert not cfg2.validation_report


def test_astengine_run_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  engine = ASTEngine(source="torch", target="stablehlo")
  code: str = "def my_func(): pass"
  import libcst as cst

  monkeypatch.setattr(ml_switcheroo.core.engine, "ingest_code", lambda *args: cst.parse_module("def foo(): pass"))

  class MockEmitter:
    """Docstring."""

    def __init__(self, semantics: typing.Any) -> None:
      """Docstring."""
      pass

    def convert(self, tree: typing.Any) -> typing.Any:
      """Docstring."""

      class MockText:
        """Docstring."""

        def to_text(self) -> str:
          """Docstring."""
          return "mock"

      return MockText()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  engine.run(code)


# --- Merged from test_engine_gap9.py ---


def test_engine_target_torch_sharding_compiler_extra_extra() -> None:
  """Verifies the behavior of engine target PyTorch sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock(nodes=["n1"])
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


def test_engine_target_keras_sharding_compiler_extra() -> None:
  """Verifies the behavior of engine target Keras sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="keras", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock(nodes=["n1"])
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


# --- Merged from test_engine_extra.py ---


def test_engine_sharding_torch() -> None:
  """Verifies the behavior of engine sharding PyTorch."""
  config = RuntimeConfig(source_framework="jax", target_framework="torch", enable_sharding=True)
  engine = ASTEngine(semantics=SemanticsManager(), config=config)
  res: ConversionResult = engine.run("import jax.numpy as jnp\njnp.array([1])")
  assert "import torch" in res.code


# --- Merged from test_engine_gap2.py ---


def get_tracer_mock_extra() -> MagicMock:
  """Gets tracer mock."""
  m = MagicMock()
  m.export.return_value = []
  return m


def test_engine_target_torch_keras_sharding() -> None:
  """Verifies the behavior of engine target PyTorch Keras sharding."""
  engine = ASTEngine(source="sass", target="keras")
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.SassParser"),
    patch("ml_switcheroo.core.engine.SassLifter"),
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_cls = MagicMock()
    mock_cls.__name__ = "PythonBackend"
    mock_get_backend.return_value = mock_cls
    mock_cls.return_value.compile.return_value = "py code"
    engine._run_compiler_pipeline("code", get_tracer_mock())


def test_rewriter_loopback() -> None:
  """Verifies the behavior of rewriter loopback."""
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map: dict[str, typing.Any] = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = [1]
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_rewriter_loopback_sharding_jax() -> None:
  """Verifies the behavior of rewriter loopback sharding JAX."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map: dict[str, typing.Any] = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = False
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_rewriter_loopback_sharding_torch() -> None:
  """Verifies the behavior of rewriter loopback sharding PyTorch."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map: dict[str, typing.Any] = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = False
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_engine_target_rdna() -> None:
  """Verifies the behavior of engine target RDNA."""
  engine = ASTEngine(source="rdna", target="keras")
  with (
    patch("ml_switcheroo.core.engine.RdnaParser"),
    patch("ml_switcheroo.core.engine.RdnaLifter"),
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_cls = MagicMock()
    mock_cls.__name__ = "PythonBackend"
    mock_get_backend.return_value = mock_cls
    mock_cls.return_value.compile.return_value = "py code"
    engine._run_compiler_pipeline("code", get_tracer_mock())


# --- Merged from test_engine_gap6.py ---


def test_engine_target_torch_sharding_rewriter_3() -> None:
  """Verifies the behavior of engine target PyTorch sharding rewriter 3."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=None):
        with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
          MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
          with patch("ml_switcheroo.core.graph.GraphExtractor") as MockExtCls:
            MockExt = MockExtCls.return_value
            g = MagicMock()
            g.nodes = ["n1"]
            MockExt.graph = g
            MockExt.node_map = {}
            engine._run_rewriter_pipeline(code, MagicMock())
            MockSharding.assert_called()


# --- Merged from test_engine_gap7.py ---


def test_engine_target_torch_sharding_compiler_extra_extra_extra() -> None:
  """Verifies the behavior of engine target PyTorch sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock()
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


def test_engine_target_torch_sharding_rewriter_4() -> None:
  """Verifies the behavior of engine target PyTorch sharding rewriter 4."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code: str = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=None):
        with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
          MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
          with patch("ml_switcheroo.core.graph.GraphExtractor") as MockExtCls:
            MockExt = MockExtCls.return_value
            g = MagicMock()
            g.nodes = ["n1"]
            MockExt.graph = g
            MockExt.node_map = {}
            engine._run_rewriter_pipeline(code, MagicMock())
            MockSharding.assert_called()


# --- Merged from test_engine_gap3.py ---


def get_tracer_mock_extra_extra() -> MagicMock:
  """Gets tracer mock."""
  m = MagicMock()
  m.export.return_value = []
  return m


def test_rewriter_loopback_sharding_full() -> None:
  """Verifies the behavior of rewriter loopback sharding full."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
    patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass"),
    patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass"),
    patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass"),
    patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass"),
  ):
    mock_differ.return_value.diff.return_value = [1]
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
      mock_extractor.return_value.node_map = {}
      engine._run_rewriter_pipeline("code", get_tracer_mock())
  engine_torch = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  engine_torch.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
    patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass"),
    patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass"),
    patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass"),
    patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass"),
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass"),
  ):
    mock_differ.return_value.diff.return_value = [1]
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
      mock_extractor.return_value.node_map = {}
      engine_torch._run_rewriter_pipeline("code", get_tracer_mock())
