"""Test suite for engine."""

from ml_switcheroo.core.compiler.ir import LogicalGraph

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
  """Verify ast engine init behavior."""
  engine = ASTEngine()
  assert isinstance(engine.semantics, SemanticsManager)
  assert engine.config is not None


def test_ast_engine_init_with_config() -> None:
  """Verify ast engine init with config behavior."""
  config = RuntimeConfig.load(source="torch", target="jax", strict_mode=True, intermediate="python")
  engine = ASTEngine(config=config, plugin_config={"a": 1}, intermediate="python")
  assert engine.strict_mode is True
  assert engine.source == "torch"
  assert engine.target == "jax"


def test_ast_engine_parse_to_source() -> None:
  """Verify ast engine parse to source behavior."""
  engine = ASTEngine()
  code = "x = 1"
  tree = engine.parse(code)
  assert isinstance(tree, cst.Module)
  assert engine.to_source(tree) == code


def test_ast_engine_graph_to_mermaid() -> None:
  """Verify ast engine graph to mermaid behavior."""
  engine = ASTEngine()
  code = "x = 1"
  tree = engine.parse(code)
  mermaid = engine._graph_to_mermaid(tree)
  assert isinstance(mermaid, str)
  assert "graph TD" in mermaid


@patch("ml_switcheroo.core.engine.ingest_code")
def test_ast_engine_run_stablehlo(mock_ingest: MagicMock) -> None:
  """Verify ast engine run stablehlo behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  mock_ingest.return_value = cst.parse_module("a = 1")
  engine = ASTEngine(source="torch", target="stablehlo")
  with patch("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter.convert") as mock_convert:
    mock_convert.return_value.to_text.return_value = "mlir_code"
    result = engine.run("a = 1")
    assert result.success is True
    assert result.code == "mlir_code"


def test_ast_engine_run_exception() -> None:
  """Verify ast engine run exception behavior."""
  engine = ASTEngine(source="torch", target="jax")
  with patch.object(engine, "_run_rewriter_pipeline", side_effect=ValueError("Test Error")):
    result = engine.run("a = 1")
    assert result.success is False
    assert "Test Error" in result.errors[0]


@patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline")
def test_ast_engine_run_compiler(mock_compiler: MagicMock) -> None:
  """Verify ast engine run compiler behavior.

  Args:
      mock_compiler: Test argument or fixture.
  """
  mock_compiler.return_value = ConversionResult(code="comp_code", success=True, trace_events=[])
  engine = ASTEngine(source="nvidia_sass", target="rdna")
  result = engine.run("some code")
  assert result.success is True
  assert result.code == "comp_code"


@patch("ml_switcheroo.core.engine.ASTEngine._run_rewriter_pipeline")
def test_ast_engine_run_rewriter(mock_rewriter: MagicMock) -> None:
  """Verify ast engine run rewriter behavior.

  Args:
      mock_rewriter: Test argument or fixture.
  """
  mock_rewriter.return_value = ConversionResult(code="rew_code", success=True, trace_events=[])
  engine = ASTEngine(source="torch", target="jax")
  result = engine.run("some code")
  assert result.success is True
  assert result.code == "rew_code"


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.PythonFrontend.parse_to_graph")
def test_run_compiler_pipeline_basic(mock_parse: MagicMock, mock_get_backend: MagicMock) -> None:
  """Verify run compiler pipeline basic behavior.

  Args:
      mock_parse: Test argument or fixture.
      mock_get_backend: Test argument or fixture.
  """
  mock_parse.return_value = MagicMock()
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled_output"
  # Setting backend_cls.__name__ to "PythonBackend"
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "PythonBackend"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="torch", target="nvidia_sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result = engine._run_compiler_pipeline("x = 1", get_tracer())
  assert result.code == "compiled_output"


@patch("ml_switcheroo.core.engine.NvidiaSassParser")
@patch("ml_switcheroo.core.engine.NvidiaSassLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_sass(mock_get_backend: MagicMock, mock_lifter: MagicMock, mock_parser: MagicMock) -> None:
  """Verify run compiler pipeline sass behavior.

  Args:
      mock_get_backend: Test argument or fixture.
      mock_lifter: Test argument or fixture.
      mock_parser: Test argument or fixture.
  """
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled_sass"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="nvidia_sass", target="rdna")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result = engine._run_compiler_pipeline("code", get_tracer())
  assert result.code == "compiled_sass"


@patch("ml_switcheroo.core.engine.RdnaParser")
@patch("ml_switcheroo.core.engine.RdnaLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_rdna(mock_get_backend: MagicMock, mock_lifter: MagicMock, mock_parser: MagicMock) -> None:
  """Verify run compiler pipeline rdna behavior.

  Args:
      mock_get_backend: Test argument or fixture.
      mock_lifter: Test argument or fixture.
      mock_parser: Test argument or fixture.
  """
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled_rdna"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="rdna", target="nvidia_sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  result = engine._run_compiler_pipeline("code", get_tracer())
  assert result.code == "compiled_rdna"


@patch("ml_switcheroo.core.engine.get_backend_class", return_value=None)
def test_run_compiler_pipeline_no_backend(mock_get: MagicMock) -> None:
  """Verify run compiler pipeline no backend behavior.

  Args:
      mock_get: Test argument or fixture.
  """
  engine = ASTEngine(source="nvidia_sass", target="jax")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with pytest.raises(ValueError, match="No backend found"):
    engine._run_compiler_pipeline("code", get_tracer())


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=True)
def test_run_compiler_pipeline_no_frontend(mock_isa: MagicMock) -> None:
  """Verify run compiler pipeline no frontend behavior.

  Args:
      mock_isa: Test argument or fixture.
  """
  engine = ASTEngine(source="html", target="nvidia_sass")
  with patch("ml_switcheroo.core.tracer.get_tracer") as mock_tracer:
    with pytest.raises(NotImplementedError):
      engine._run_compiler_pipeline("code", mock_tracer())


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_compiler_pipeline_mlir_ingest(mock_ingest: MagicMock, mock_get_backend: MagicMock) -> None:
  """Verify run compiler pipeline mlir ingest behavior.

  Args:
      mock_ingest: Test argument or fixture.
      mock_get_backend: Test argument or fixture.
  """
  mock_ingest.return_value = cst.parse_module("x = 1")
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="mlir", target="nvidia_sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  res = engine._run_compiler_pipeline("code", get_tracer())
  assert res.success is True
  assert res.code == "compiled"


@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_compiler_pipeline_mlir_ingest_fallback(mock_ingest: MagicMock, mock_get_backend: MagicMock) -> None:
  """Verify run compiler pipeline mlir ingest fallback behavior.

  Args:
      mock_ingest: Test argument or fixture.
      mock_get_backend: Test argument or fixture.
  """
  # Mock ingest to raise exception to trigger fallback
  mock_ingest.side_effect = RuntimeError("parse error")
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="mlir", target="nvidia_sass")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  # Needs valid python to pass PythonFrontend inside fallback
  res = engine._run_compiler_pipeline("x = 1", get_tracer())
  assert res.success is True
  assert res.code == "compiled"


@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_optimizations(mock_get_backend: MagicMock) -> None:
  """Verify run compiler pipeline optimizations behavior.

  Args:
      mock_get_backend: Test argument or fixture.
  """
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
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
                    res = engine._run_compiler_pipeline("x = 1", get_tracer())
  assert res.success is True
  assert res.code == "compiled"


@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_optimizations_else(mock_get_backend: MagicMock) -> None:
  """Verify run compiler pipeline optimizations else behavior.

  Args:
      mock_get_backend: Test argument or fixture.
  """
  mock_backend_cls = MagicMock()
  mock_backend_cls.__name__ = "OtherBackend"
  mock_backend = MagicMock()
  mock_backend.compile.return_value = "compiled"
  mock_backend_cls.return_value = mock_backend
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="jax", target="torch", enable_graph_optimization=True)
  engine.config.enable_sharding = True

  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize", return_value=MagicMock()):
    with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=MagicMock()):
      res = engine._run_compiler_pipeline("x = 1", get_tracer())
  assert res.success is True
  assert res.code == "compiled"


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_basic(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline basic behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "import torch\nx = 1\n# <SWITCHEROO_ESCAPE>\nx=2"
  mock_ingest.return_value = cst.parse_module(code)

  engine = ASTEngine(source="torch", target="jax")
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.core.engine.EscapeHatch") as mock_hatch:
      mock_hatch.START_MARKER = "# <SWITCHEROO_ESCAPE>"
      result = engine._run_rewriter_pipeline(code, get_tracer())

  assert result.success is True
  assert any("Escape Hatches Detected" in err for err in result.errors)


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_no_import_fixer(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline no import fixer behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax")
  engine.config.enable_import_fixer = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    result = engine._run_rewriter_pipeline(code, get_tracer())
  assert result.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_strict_no_errors(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline strict no errors behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax", strict_mode=True)
  engine.config.enable_import_fixer = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.testing.linter.StructuralLinter.check", return_value=[]):
      result = engine._run_rewriter_pipeline(code, get_tracer())
  assert result.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_optimizations(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline optimizations behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)

  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True, strict_mode=True)
  engine.config.enable_sharding = True
  engine.config.enable_import_fixer = True
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  mock_graph = MagicMock()
  mock_graph.nodes = ["mock_node"]

  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor_class:
    mock_extractor_instance = MagicMock()
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
                                res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


def test_run_rewriter_pipeline_optimizations_exception() -> None:
  """Verify run rewriter pipeline optimizations exception behavior."""
  code = "x = 1"
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module(code)):
    with patch("ml_switcheroo.core.engine.GraphExtractor", side_effect=RuntimeError("Extraction Failed")):
      res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


def test_ast_engine_validation_report() -> None:
  """Verify ast engine validation report behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", validation_report=Path("foo.json"))
  mock_semantics = MagicMock()
  ASTEngine(config=config, semantics=mock_semantics)
  mock_semantics.load_validation_report.assert_called_once_with(Path("foo.json"))


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_nodes(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline opt no nodes behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = ""
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
    res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_plan_1(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline opt no plan 1 behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()

  def mock_diff(*args: LogicalGraph, **kwargs: LogicalGraph) -> list[str]:
    """Execute mock diff helper logic.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return []

  with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", side_effect=mock_diff):
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize") as mock_opt:
      mock_opt.return_value = "fake_optimized"
      with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
        res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_sharding(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline opt no sharding behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst = MagicMock()
    mock_inst.graph.nodes = ["node"]
    mock_ext.return_value = mock_inst
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=["plan"]):
        with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"):
          with patch("libcst.Module.visit", return_value=cst.parse_module(code)):
            with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
              res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_sharding_not_jax(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline opt sharding not jax behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="torch", enable_graph_optimization=True)
  engine.config.enable_sharding = True
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst = MagicMock()
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
                        res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
def test_run_rewriter_pipeline_opt_no_plan(mock_ingest: MagicMock) -> None:
  """Verify run rewriter pipeline opt no plan behavior.

  Args:
      mock_ingest: Test argument or fixture.
  """
  code = "x = 1"
  mock_ingest.return_value = cst.parse_module(code)
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = False
  from ml_switcheroo.core.tracer import get_tracer, reset_tracer

  reset_tracer()
  with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_ext:
    mock_inst = MagicMock()
    mock_inst.graph.nodes = ["node"]
    mock_ext.return_value = mock_inst
    with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=[]):  # Empty plan
        with patch("ml_switcheroo.core.rewriter.pipeline.RewriterPipeline.run", return_value=cst.parse_module(code)):
          res = engine._run_rewriter_pipeline(code, get_tracer())
  assert res.success is True


class DummyMixin3(AttributeMixin):
  """Dummy mixin 3."""

  pass


def test_attributes_mixin_branch_coverage3() -> None:
  """Verify attributes mixin branch coverage3 behavior."""
  mixin = DummyMixin3()
  # Let's hit the lines without `_path_to_alias` and without `_defined_names`
  node = cst.ensure_type(cst.parse_expression("a.b"), cst.Attribute)
  assert mixin.leave_Attribute(node, node) == node

  node2 = cst.ensure_type(cst.parse_expression("a.module.c"), cst.Attribute)
  assert mixin._simplify_reexports(node2) == node2


def test_engine_compiler_safety_warning_logging() -> None:
  """Verify that compiler pipeline logs warnings for detected static safety diagnostics."""
  from ml_switcheroo.config import RuntimeConfig

  cfg = RuntimeConfig.load(source="torch", target="rdna")
  engine = ASTEngine(config=cfg)
  code = "import torch\ndef f(x):\n    if x.item() > 0:\n        return x\n"
  result = engine.run(code)
  assert any("VALUE_DEPENDENT_CONTROL_FLOW" in w for w in result.warnings)


def test_engine_rewriter_safety_warning_logging() -> None:
  """Verify that rewriter pipeline logs warnings for detected static safety diagnostics."""
  from ml_switcheroo.config import RuntimeConfig

  cfg = RuntimeConfig.load(source="torch", target="jax")
  engine = ASTEngine(config=cfg)
  code = "import torch\ndef f(x):\n    if x.item() > 0:\n        return x\n"
  result = engine.run(code)
  assert any("VALUE_DEPENDENT_CONTROL_FLOW" in w for w in result.warnings)


def test_engine_safety_scanner_exception_handling() -> None:
  """Verify graceful exception handling when StaticSafetyScanner encounters an unexpected error."""
  from ml_switcheroo.analysis.static_safety import StaticSafetyScanner
  from ml_switcheroo.config import RuntimeConfig

  # Compiler pipeline exception handling
  cfg_comp = RuntimeConfig.load(source="torch", target="rdna")
  engine_comp = ASTEngine(config=cfg_comp)
  with patch.object(StaticSafetyScanner, "scan", side_effect=RuntimeError("Scanner error")):
    res_comp = engine_comp.run("import torch\nx = 1\n")
    assert res_comp.success is True

  # Rewriter pipeline exception handling
  cfg_rew = RuntimeConfig.load(source="torch", target="jax")
  engine_rew = ASTEngine(config=cfg_rew)
  with patch.object(StaticSafetyScanner, "scan", side_effect=RuntimeError("Scanner error")):
    res_rew = engine_rew.run("import torch\nx = 1\n")
    assert res_rew.success is True


def test_engine_transpile_to_ir_json_and_python() -> None:
  """Verifies that transpiling to 'ir' produces JSON and 'ml_switcheroo_ir' produces Python CST with imports."""
  import json

  from ml_switcheroo.config import RuntimeConfig

  torch_code = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class SampleNet(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.linear = nn.Linear(4, 8)\n"
    "    def forward(self, x):\n"
    "        return self.linear(x)\n"
  )

  # Target 'ir' -> JSON output
  cfg_ir = RuntimeConfig.load(source="torch", target="ir")
  eng_ir = ASTEngine(config=cfg_ir)
  res_ir = eng_ir.run(torch_code)
  assert res_ir.success is True
  json_data = json.loads(res_ir.code)
  assert json_data["name"] == "SampleNet"
  assert "nodes" in json_data
  assert "edges" in json_data

  # Target 'ml_switcheroo_ir' -> Python code with import ml_switcheroo_ir as sw_ir
  cfg_sw = RuntimeConfig.load(source="torch", target="ml_switcheroo_ir")
  eng_sw = ASTEngine(config=cfg_sw)
  res_sw = eng_sw.run(torch_code)
  assert res_sw.success is True
  assert "import ml_switcheroo_ir as sw_ir" in res_sw.code
  assert "def build_graph() -> sw_ir.LogicalGraph:" in res_sw.code
  assert "name='SampleNet'" in res_sw.code


def test_engine_consolidated_core_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify consolidated ASTEngine branches from tests/core/test_engine.py.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.core.engine

  # 1. Intermediate config initialization
  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine1 = ASTEngine(config=cfg1)
  assert engine1.config.intermediate == "mlir"

  engine2 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine2.config.intermediate == "onnx"

  # 2. Sharding run via run()
  with patch("ml_switcheroo.core.engine.ASTEngine._run_compiler_pipeline") as mock_run_compiler:
    mock_run_compiler.return_value = ConversionResult(success=True)
    cfg_shard = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
    engine_shard = ASTEngine(config=cfg_shard)
    res_shard = engine_shard.run("code")
    assert res_shard.success is True
    mock_run_compiler.assert_called_once()

  # 3. StableHLO with get_adapter returning None
  engine_shlo = ASTEngine(source="torch", target="stablehlo")
  with patch("ml_switcheroo.core.engine.get_adapter", return_value=None):
    res_shlo = engine_shlo.run("def f(): pass")
    assert res_shlo.success is True

  # 4. Compiler pipeline with fusion targets (flax, flax_nnx, paxml, torch)
  class FakeBackend:
    """Test helper class FakeBackend."""

    def __init__(self, semantics: SemanticsManager) -> None:
      """Initialize test helper instance.

      Args:
          semantics: Test argument or fixture.
      """
      pass

    def compile(self, graph: LogicalGraph) -> str:
      """Return compiled string.

      Args:
          graph: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "COMP"

  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: FakeBackend)
  cfg_fusion = RuntimeConfig(
    source_framework="nvidia_sass", target_framework="jax", enable_graph_optimization=True, enable_sharding=True
  )
  engine_fusion = ASTEngine(config=cfg_fusion)
  for target_fw in ("flax", "flax_nnx", "paxml", "torch", "unknown_fw"):
    engine_fusion.target = target_fw
    res_f = engine_fusion.run("FADD R0, R1, R2;")
    assert res_f.success is True

  engine_fusion.target = "non_dict_conf"
  monkeypatch.setattr(engine_fusion.semantics, "get_framework_config", lambda fw: {})
  assert engine_fusion._is_sharding_fusion_target() is False

  # 5. Rewriter pipeline with target == "mlir"
  engine_mlir = ASTEngine(source="torch", target="mlir")
  tracer = MagicMock()
  tracer.export.return_value = []
  res_mlir = engine_mlir._run_rewriter_pipeline("def foo(): pass", tracer)
  assert res_mlir.success is True

  # 6. PipelineStrategy base methods and empty strategy registry
  from ml_switcheroo.core.strategies import PipelineStrategy

  base_strat = PipelineStrategy()
  with pytest.raises(NotImplementedError):
    base_strat.can_handle(engine_mlir)
  with pytest.raises(NotImplementedError):
    base_strat.execute(engine_mlir, "code", tracer)

  with patch("ml_switcheroo.core.strategies.STRATEGY_REGISTRY", []):
    res_empty = engine_mlir.run("x = 1")
    assert res_empty.success is False
