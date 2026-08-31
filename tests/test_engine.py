"""Module docstring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeMixin
from ml_switcheroo.semantics.manager import SemanticsManager


def test_ast_engine_init() -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine()
  assert isinstance(engine.semantics, SemanticsManager)
  assert engine.config is not None


def test_ast_engine_init_with_config() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig.load(source="torch", target="jax", strict_mode=True, intermediate="python")
  engine: ASTEngine = ASTEngine(config=config, plugin_config={"a": 1}, intermediate="python")
  assert engine.strict_mode is True
  assert engine.source == "torch"
  assert engine.target == "jax"


def test_ast_engine_parse_to_source() -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine()
  code: str = "x = 1"
  tree: cst.Module = engine.parse(code)
  assert isinstance(tree, cst.Module)
  assert engine.to_source(tree) == code


def test_ast_engine_graph_to_mermaid() -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine()
  code: str = "x = 1"
  tree: cst.Module = engine.parse(code)
  mermaid: str = engine._graph_to_mermaid(tree)
  assert isinstance(mermaid, str)
  assert "graph TD" in mermaid


@patch("ml_switcheroo.core.engine.ingest_code")
def test_ast_engine_run_stablehlo(mock_ingest: MagicMock) -> None:
  """Docstring."""
  mock_ingest.return_value = cst.parse_module("a = 1")
  engine: ASTEngine = ASTEngine(source="torch", target="stablehlo")
  with patch("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter.convert") as mock_convert:
    mock_convert.return_value.to_text.return_value = "mlir_code"
    result: ConversionResult = engine.run("a = 1")
    assert result.success is True
    assert result.code == "mlir_code"


def test_ast_engine_run_exception() -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine(source="torch", target="jax")
  with patch.object(engine, "_run_rewriter_pipeline", side_effect=ValueError("Test Error")):
    result: ConversionResult = engine.run("a = 1")
    assert result.success is False
    assert "Test Error" in result.errors[0]


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_ast_engine_run_compiler(mock_compiler: MagicMock) -> None:
  """Docstring."""
  mock_compiler.return_value = ConversionResult(code="comp_code", success=True, trace_events=[])
  engine: ASTEngine = ASTEngine(source="sass", target="rdna")
  result: ConversionResult = engine.run("some code")
  assert result.success is True
  assert result.code == "comp_code"


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_ast_engine_run_rewriter(mock_rewriter: MagicMock) -> None:
  """Docstring."""
  mock_rewriter.return_value = ConversionResult(code="rew_code", success=True, trace_events=[])
  engine: ASTEngine = ASTEngine(source="torch", target="jax")
  result: ConversionResult = engine.run("some code")
  assert result.success is True
  assert result.code == "rew_code"


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.PythonFrontend.parse_to_graph")
def test_run_compiler_pipeline_basic(mock_parse: MagicMock, mock_get_backend: MagicMock) -> None:
  """Docstring."""
  mock_parse.return_value = MagicMock()
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled_output"
  # Setting backend_cls.__name__ to "PythonBackend"
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "PythonBackend"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="torch", target="sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result: ConversionResult = engine._run_compiler_pipeline("x = 1", get_tracer())
  assert result.code == "compiled_output"


@patch("ml_switcheroo.core.engine.SassParser")
@patch("ml_switcheroo.core.engine.SassLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_sass(mock_get_backend: MagicMock, mock_lifter: MagicMock, mock_parser: MagicMock) -> None:
  """Docstring."""
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled_sass"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="sass", target="rdna")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result: ConversionResult = engine._run_compiler_pipeline("code", get_tracer())
  assert result.code == "compiled_sass"


@patch("ml_switcheroo.core.engine.RdnaParser")
@patch("ml_switcheroo.core.engine.RdnaLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_rdna(mock_get_backend: MagicMock, mock_lifter: MagicMock, mock_parser: MagicMock) -> None:
  """Docstring."""
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled_rdna"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="rdna", target="sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result: ConversionResult = engine._run_compiler_pipeline("code", get_tracer())
  assert result.code == "compiled_rdna"


@patch("ml_switcheroo.core.engine.get_backend_class", return_value=None)
def test_run_compiler_pipeline_no_backend(mock_get: MagicMock) -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine(source="sass", target="jax")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with pytest.raises(ValueError, match="No backend found"):
    engine._run_compiler_pipeline("code", get_tracer())


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=True)
def test_run_compiler_pipeline_no_frontend(mock_isa: MagicMock) -> None:
  """Docstring."""
  engine: ASTEngine = ASTEngine(source="html", target="sass")
  with patch("ml_switcheroo.core.tracer.get_tracer") as mock_tracer:
    with pytest.raises(NotImplementedError):
      engine._run_compiler_pipeline("code", mock_tracer())


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_compiler_pipeline_mlir_ingest(mock_ingest: MagicMock, mock_get_backend: MagicMock) -> None:
  """Docstring."""
  mock_ingest.return_value = cst.parse_module("x = 1")
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="mlir", target="sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  engine._run_compiler_pipeline("code", get_tracer())


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_compiler_pipeline_mlir_ingest_fallback(mock_ingest: MagicMock, mock_get_backend: MagicMock) -> None:
  """Docstring."""
  # Mock ingest to raise exception to trigger fallback
  mock_ingest.side_effect = Exception("parse error")
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="mlir", target="sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  # Needs valid python to pass PythonFrontend inside fallback
  engine._run_compiler_pipeline("x = 1", get_tracer())


@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_optimizations(mock_get_backend: MagicMock) -> None:
  """Docstring."""
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = True

  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  # Mock methods to allow optimization and sharding to run without crashing
  with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize", return_value=MagicMock()):
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=MagicMock()):
      with patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=MagicMock()):
        with patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=MagicMock()):
          with patch(
            "ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=MagicMock()
          ):
            with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=MagicMock()):
              with patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=MagicMock()):
                with patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=MagicMock()):
                  with patch(
                    "ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply",
                    return_value=MagicMock(),
                  ):
                    engine._run_compiler_pipeline("x = 1", get_tracer())


@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_optimizations_else(mock_get_backend: MagicMock) -> None:
  """Docstring."""
  mock_backend_cls: MagicMock = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend: MagicMock = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine: ASTEngine = ASTEngine(source="jax", target="torch", enable_graph_optimization=True)
  engine.config.enable_sharding = True

  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize", return_value=MagicMock()):
    with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=MagicMock()):
      engine._run_compiler_pipeline("x = 1", get_tracer())


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_basic(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "import torch\nx = 1\n# <SWITCHEROO_ESCAPE>\nx=2"
  mock_ingest.return_value = cst.parse_module(code)

  engine: ASTEngine = ASTEngine(source="torch", target="jax")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.core.engine.EscapeHatch") as mock_hatch:
      mock_hatch.START_MARKER = "# <SWITCHEROO_ESCAPE>"
      result: ConversionResult = engine._run_rewriter_pipeline(code, get_tracer())

  assert result.success is True
  assert any("Escape Hatches Detected" in err for err in result.errors)


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_no_import_fixer(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax")
  engine.config.enable_import_fixer = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    result: ConversionResult = engine._run_rewriter_pipeline(code, get_tracer())
  assert result.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_strict_no_errors(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax", strict_mode=True)
  engine.config.enable_import_fixer = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.testing.linter.StructuralLinter.check", return_value=[]):
      result: ConversionResult = engine._run_rewriter_pipeline(code, get_tracer())
  assert result.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_optimizations(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)

  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True, strict_mode=True)
  engine.config.enable_sharding = True
  engine.config.enable_import_fixer = True
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  mock_graph: MagicMock = MagicMock()
  mock_graph.nodes = ["mock_node"]

  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor_class:
    mock_extractor_instance: MagicMock = MagicMock()
    mock_extractor_instance.graph = mock_graph
    mock_extractor_instance.node_map = {}
    mock_extractor_class.return_value = mock_extractor_instance
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize", return_value=mock_graph):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=["plan"]):
        with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as _:
          with patch("libcst.Module.visit", return_value=cst.parse_module(code)):
            with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
              with patch("ml_switcheroo.testing.linter.StructuralLinter.check", return_value=["error"]):
                with patch(
                  "ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph
                ):
                  with patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph):
                    with patch(
                      "ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph
                    ):
                      with patch(
                        "ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply",
                        return_value=mock_graph,
                      ):
                        with patch(
                          "ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph
                        ):
                          with patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph):
                            with patch(
                              "ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph
                            ):
                              with patch(
                                "ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply",
                                return_value=mock_graph,
                              ):
                                engine._run_rewriter_pipeline(code, get_tracer())


def test_run_rewriter_pipeline_optimizations_exception() -> None:
  """Docstring."""
  code: str = "x = 1"
  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.core.engine.GraphExtractor", side_effect=Exception("Extraction Failed")):
      engine._run_rewriter_pipeline(code, get_tracer())


def test_ast_engine_validation_report() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(
    source_framework="torch", target_framework="jax", validation_report=Path("foo.json")
  )
  print(f"DEBUG: {config.validation_report}")

  mock_semantics: MagicMock = MagicMock()
  ASTEngine(config=config, semantics=mock_semantics)
  mock_semantics.load_validation_report.assert_called_once_with(Path("foo.json"))


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_nodes(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = ""
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    engine._run_rewriter_pipeline(code, get_tracer())


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_plan_1(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  def mock_diff(*args: list[str], **kwargs: dict[str, str]) -> list[str]:
    """Docstring."""
    print("MOCK DIFF CALLED!")
    return []

  print(f"ENGINE CONFIG GRAPH OPT = {engine.config.enable_graph_optimization}")

  with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", side_effect=mock_diff):
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize") as mock_opt:
      mock_opt.return_value = "fake_optimized"
      with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
        engine._run_rewriter_pipeline(code, get_tracer())


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_sharding(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst: MagicMock = MagicMock()
    mock_inst.graph.nodes = ["node"]
    mock_ext.return_value = mock_inst
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=["plan"]):
        with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"):
          with patch("libcst.Module.visit", return_value=cst.parse_module(code)):
            with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
              engine._run_rewriter_pipeline(code, get_tracer())


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_sharding_not_jax(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="torch", enable_graph_optimization=True)
  engine.config.enable_sharding = True
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst: MagicMock = MagicMock()
    mock_inst.graph.nodes = ["node"]
    mock_ext.return_value = mock_inst
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=["plan"]):
        with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"):
          with patch("libcst.Module.visit", return_value=cst.parse_module(code)):
            with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
              with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
                with patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply"):
                  with patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply"):
                    with patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply"):
                      with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply"):
                        engine._run_rewriter_pipeline(code, get_tracer())


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_plan(mock_ingest: MagicMock) -> None:
  """Docstring."""
  code: str = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine: ASTEngine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst: MagicMock = MagicMock()
    mock_inst.graph.nodes = ["node"]
    mock_ext.return_value = mock_inst
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=[]):  # Empty plan
        with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
          engine._run_rewriter_pipeline(code, get_tracer())
          print("PIPELINE RAN")


class DummyMixin3(AttributeMixin):
  """Dummy mixin 3."""

  pass


def test_attributes_mixin_branch_coverage3() -> None:
  """Docstring."""
  mixin: DummyMixin3 = DummyMixin3()
  # Let's hit the lines without `_path_to_alias` and without `_defined_names`
  node: cst.BaseExpression = cst.parse_expression("a.b")
  assert mixin.leave_Attribute(node, node) == node

  node2: cst.BaseExpression = cst.parse_expression("a.module.c")
  assert mixin._simplify_reexports(node2) == node2
