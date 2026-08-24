"""Test module."""

import pytest
import libcst as cst
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.graph import LogicalGraph


@patch("ml_switcheroo.core.engine.PythonFrontend")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_engine_run_compiler_pipeline_python(mock_get_backend, mock_frontend_class):
  """Test element."""
  mock_frontend = mock_frontend_class.return_value
  mock_graph = LogicalGraph()
  mock_frontend.parse_to_graph.return_value = mock_graph

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "PythonBackend"
  mock_backend = mock_backend_class.return_value
  mock_backend.compile.return_value = "compiled_python"
  mock_get_backend.return_value = mock_backend_class

  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
  engine = ASTEngine(config=config)  # Forces compiler pipeline via enable_sharding

  # We mock ingest_code inside engine to just return a basic CST to avoid complex python parsing
  with patch("ml_switcheroo.core.engine.ingest_code") as mock_ingest:
    mock_ingest.return_value = cst.parse_module("code")
    res = engine._run_compiler_pipeline("code", MagicMock())

    assert res.success is True
    assert res.code == "compiled_python"


@patch("ml_switcheroo.core.engine.SassParser")
@patch("ml_switcheroo.core.engine.SassLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_engine_run_compiler_pipeline_sass(mock_get_backend, mock_lifter_class, mock_parser_class):
  """Test element."""
  mock_parser = mock_parser_class.return_value
  mock_parser.parse.return_value.statements = []

  mock_lifter = mock_lifter_class.return_value
  mock_lifter.lift.return_value = LogicalGraph()

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "OtherBackend"
  mock_get_backend.return_value = mock_backend_class
  mock_backend_class.return_value.compile.return_value = "compiled_sass"

  engine = ASTEngine(source="sass", target="html")
  res = engine._run_compiler_pipeline("code", MagicMock())

  assert res.success is True
  assert res.code == "compiled_sass"


@patch("ml_switcheroo.core.engine.RdnaParser")
@patch("ml_switcheroo.core.engine.RdnaLifter")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_engine_run_compiler_pipeline_rdna(mock_get_backend, mock_lifter_class, mock_parser_class):
  """Test element."""
  mock_parser = mock_parser_class.return_value
  mock_parser.parse.return_value.statements = []

  mock_lifter = mock_lifter_class.return_value
  mock_lifter.lift.return_value = LogicalGraph()

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "OtherBackend"
  mock_get_backend.return_value = mock_backend_class
  mock_backend_class.return_value.compile.return_value = "compiled_rdna"

  engine = ASTEngine(source="rdna", target="html")
  res = engine._run_compiler_pipeline("code", MagicMock())

  assert res.success is True
  assert res.code == "compiled_rdna"


@patch("ml_switcheroo.core.engine.PythonFrontend")
@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.engine.ingest_code")
def test_engine_run_compiler_pipeline_stablehlo(mock_ingest, mock_get_backend, mock_frontend_class):
  """Test element."""
  mock_ingest.return_value = cst.parse_module("code")
  mock_frontend = mock_frontend_class.return_value
  mock_frontend.parse_to_graph.return_value = LogicalGraph()

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "OtherBackend"
  mock_backend_class.return_value.compile.return_value = "compiled_stablehlo"
  mock_get_backend.return_value = mock_backend_class

  engine = ASTEngine(source="stablehlo", target="html")
  res = engine._run_compiler_pipeline("code", MagicMock())

  assert res.success is True
  assert res.code == "compiled_stablehlo"


@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
@patch("ml_switcheroo.core.engine.UsageScanner")
@patch("ml_switcheroo.core.engine.ImportResolver")
@patch("ml_switcheroo.core.engine.ImportFixer")
def test_engine_run_rewriter_pipeline(mock_fixer, mock_resolver, mock_scanner, mock_pipeline_class, mock_ingest):
  """Test element."""
  mock_ingest.return_value = cst.parse_module("code")

  mock_pipeline = mock_pipeline_class.return_value
  mock_pipeline.run.return_value = cst.parse_module("rewritten_code")

  mock_scanner_inst = mock_scanner.return_value
  mock_scanner_inst.get_result.return_value = True

  mock_resolver_inst = mock_resolver.return_value
  mock_resolver_inst.resolve.return_value = []

  mock_fixer_inst = mock_fixer.return_value
  mock_fixer_inst.visit = lambda node: cst.parse_module("fixed_code")

  # Needs to ensure visit returns something
  class MockTree:
    code = "final_code"

    def visit(self, visitor):
      if type(visitor).__name__ == "UsageScanner":
        return self
      return self

  mock_pipeline.run.return_value = MockTree()

  engine = ASTEngine(source="torch", target="jax", strict_mode=True)  # strict mode on to trigger linter

  with patch("ml_switcheroo.core.engine.StructuralLinter") as mock_linter_class:
    mock_linter = mock_linter_class.return_value
    mock_linter.check.return_value = ["linter_error"]

    res = engine._run_rewriter_pipeline("code", MagicMock())

    assert res.success is True
    assert res.code == "final_code"
    assert "linter_error" in res.errors


@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
@patch("ml_switcheroo.core.engine.GraphExtractor")
@patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer")
@patch("ml_switcheroo.core.compiler.differ.GraphDiffer")
@patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher")
@patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter")
def test_engine_run_rewriter_pipeline_graph_opt(
  mock_emitter, mock_patcher, mock_differ, mock_opt, mock_extractor, mock_pipeline, mock_ingest
):
  """Test element."""
  mock_ingest.return_value = cst.parse_module("code")

  mock_extractor_inst = mock_extractor.return_value
  mock_extractor_inst.graph = LogicalGraph()
  # Add a dummy node to trigger the "if original_graph.nodes:" branch
  from ml_switcheroo.core.compiler.ir import LogicalNode

  mock_extractor_inst.graph.nodes = {"n1": LogicalNode(id="n1", kind="dummy")}
  mock_extractor_inst.node_map = {}

  mock_opt_inst = mock_opt.return_value
  mock_opt_inst.optimize.return_value = LogicalGraph()

  mock_differ_inst = mock_differ.return_value
  mock_differ_inst.diff.return_value = ["plan"]  # Some dummy plan

  class MockTree:
    code = "final"

    def visit(self, visitor):
      return self

  mock_ingest.return_value = MockTree()
  mock_pipeline.return_value.run.return_value = MockTree()

  config = RuntimeConfig(
    source_framework="torch", target_framework="jax", enable_graph_optimization=True, enable_sharding=True
  )
  engine = ASTEngine(config=config)

  # We need to patch the sharding passes to just return the graph
  with (
    patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass") as me,
    patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass") as md1,
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass") as md2,
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass") as md3,
    patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass") as mi,
    patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass") as mf1,
    patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass") as mf2,
    patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass") as mf3,
  ):
    for mock_pass in [me, md1, md2, md3, mi, mf1, mf2, mf3]:
      mock_pass.return_value.apply.side_effect = lambda g: g

    res = engine._run_rewriter_pipeline("code", MagicMock())
    assert res.success is True


def test_engine_run_compiler_pipeline_unknown_isa():
  """Test element."""
  # We can't use an unknown ISA in ASTEngine initialization because RuntimeConfig validates it.
  # So we bypass the validation by patching is_isa_source locally in the test instead
  engine = ASTEngine(source="sass", target="html")  # use valid frameworks
  engine.source = "unknown_isa"  # Override after init

  with patch("ml_switcheroo.core.engine.is_isa_source", return_value=True):
    with pytest.raises(NotImplementedError, match="No frontend for unknown_isa"):
      engine._run_compiler_pipeline("code", MagicMock())


@patch("ml_switcheroo.core.engine.PythonFrontend")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_engine_run_compiler_pipeline_no_backend(mock_get_backend, mock_frontend_class):
  """Test element."""
  mock_get_backend.return_value = None
  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
  engine = ASTEngine(config=config)
  engine.target = "unknown_backend"  # override to avoid pydantic
  with patch("ml_switcheroo.core.engine.ingest_code"):
    with pytest.raises(ValueError, match="No backend found for unknown_backend"):
      engine._run_compiler_pipeline("code", MagicMock())


@patch("ml_switcheroo.core.engine.PythonFrontend")
@patch("ml_switcheroo.core.engine.get_backend_class")
@patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer")
def test_engine_run_compiler_pipeline_graph_opt(mock_opt, mock_get_backend, mock_frontend_class):
  """Test element."""
  mock_opt_inst = mock_opt.return_value
  mock_opt_inst.optimize.return_value = LogicalGraph()

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "PythonBackend"
  mock_backend_class.return_value.compile.return_value = "code_str"
  mock_get_backend.return_value = mock_backend_class

  config = RuntimeConfig(
    source_framework="torch", target_framework="jax", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(config=config)
  with patch("ml_switcheroo.core.engine.ingest_code"):
    with (
      patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass") as me,
      patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass") as md1,
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass") as md2,
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass") as md3,
      patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass") as mi,
      patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass") as mf1,
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass") as mf2,
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass") as mf3,
    ):
      for mock_pass in [me, md1, md2, md3, mi, mf1, mf2, mf3]:
        mock_pass.return_value.apply.side_effect = lambda g: g

      engine._run_compiler_pipeline("code", MagicMock())
      mock_opt.assert_called_once()


@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
@patch("ml_switcheroo.core.engine.GraphExtractor")
@patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer")
@patch("ml_switcheroo.core.compiler.differ.GraphDiffer")
def test_engine_run_rewriter_pipeline_graph_opt_no_plan(
  mock_differ, mock_opt, mock_extractor, mock_pipeline, mock_ingest
):
  """Test element."""
  mock_differ.return_value.diff.return_value = []  # Empty plan
  mock_extractor_inst = mock_extractor.return_value
  mock_extractor_inst.graph = LogicalGraph()
  from ml_switcheroo.core.compiler.ir import LogicalNode

  mock_extractor_inst.graph.nodes = {"n1": LogicalNode(id="n1", kind="dummy")}
  mock_extractor_inst.node_map = {}

  class MockTree:
    code = "final"

    def visit(self, visitor):
      return self

  mock_ingest.return_value = MockTree()
  mock_pipeline.return_value.run.return_value = MockTree()

  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_graph_optimization=True)
  engine = ASTEngine(config=config)
  res = engine._run_rewriter_pipeline("code", MagicMock())
  assert res.success is True


@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
@patch("ml_switcheroo.core.engine.GraphExtractor")
def test_engine_run_rewriter_pipeline_graph_opt_fail(mock_extractor, mock_pipeline, mock_ingest):
  """Test element."""
  mock_extractor.side_effect = Exception("Graph Extraction Failed")

  class MockTree:
    code = "final"

    def visit(self, visitor):
      return self

  mock_ingest.return_value = MockTree()
  mock_pipeline.return_value.run.return_value = MockTree()

  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_graph_optimization=True)
  engine = ASTEngine(config=config)
  res = engine._run_rewriter_pipeline("code", MagicMock())
  assert res.success is True  # Should proceed with raw CST
  assert len(res.errors) == 0


@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
def test_engine_run_rewriter_pipeline_escape_hatch(mock_pipeline, mock_ingest):
  """Test element."""

  class MockTree:
    # Needs to exactly match the start marker string
    code = "# <SWITCHEROO_FAIL_123"

    def visit(self, visitor):
      return self

  mock_ingest.return_value = MockTree()
  mock_pipeline.return_value.run.return_value = MockTree()

  engine = ASTEngine(source="torch", target="jax")

  # We must patch EscapeHatch.START_MARKER to match our code
  with patch("ml_switcheroo.core.engine.EscapeHatch.START_MARKER", "# <SWITCHEROO_FAIL_"):
    res = engine._run_rewriter_pipeline("code", MagicMock())
    assert len(res.errors) > 0
    assert "Escape Hatches Detected" in res.errors[0]


@patch("ml_switcheroo.core.engine.PythonFrontend")
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_engine_run_compiler_pipeline_ingest_fallback(mock_get_backend, mock_frontend_class):
  """Test element."""
  mock_frontend = mock_frontend_class.return_value
  mock_graph = LogicalGraph()
  mock_frontend.parse_to_graph.return_value = mock_graph

  mock_backend_class = MagicMock()
  mock_backend_class.__name__ = "PythonBackend"
  mock_backend = mock_backend_class.return_value
  mock_backend.compile.return_value = "compiled_python"
  mock_get_backend.return_value = mock_backend_class

  config = RuntimeConfig(source_framework="torch", target_framework="jax", enable_sharding=True)
  engine = ASTEngine(config=config)

  # Force ingest_code to raise an exception, triggering the fallback
  with patch("ml_switcheroo.core.engine.ingest_code", side_effect=Exception("Ingest Failed")):
    res = engine._run_compiler_pipeline("code", MagicMock())
    assert res.success is True
    assert res.code == "compiled_python"
