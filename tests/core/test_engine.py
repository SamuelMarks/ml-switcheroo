"""Unit tests for the ASTEngine core orchestration component.

This module validates that ASTEngine behaves correctly across various ingestion,
parsing, rewriting, linter, and compiling pipelines. This includes testing:
- Default and config-based initialization.
- Ingestion and roundtrip parsing to/from source strings.
- Executing compiler pipelines for SASS and RDNA architectures.
- Executing rewriter, linter, and StableHLO pipelines.
- Exception and unsupported-source graceful handling.
- Graph visualization generation (Mermaid).
- Proper propagation and enforcement of flags like strict_mode.
"""

from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
import libcst as cst


def test_astengine_init_defaults():
  """Verifies the default initialization properties of ASTEngine.

  This test checks that ASTEngine initializes with correct framework values
  and default options (e.g., strict_mode disabled) when config is not
  provided.

  Args:
      None

  Returns:
      None
  """
  engine = ASTEngine(source="torch", target="jax")
  assert engine.source == "torch"
  assert engine.target == "jax"
  assert engine.strict_mode is False


def test_astengine_init_with_config():
  """Verifies ASTEngine initialization behavior when specified with RuntimeConfig.

  This test checks that ASTEngine respects custom configurations supplied via
  RuntimeConfig, validating proper propagation of options like intermediate
  target formats and strict mode.

  Args:
      None

  Returns:
      None
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(config=config, intermediate="mlir")
  assert engine.config.intermediate == "mlir"
  assert engine.strict_mode is True


def test_astengine_parse_to_source():
  """Verifies ASTEngine can roundtrip parse and regenerate Python code.

  This test verifies that the `parse` and `to_source` methods correctly convert
  a source code string to a LibCST Module and back without loss.

  Args:
      None

  Returns:
      None
  """
  engine = ASTEngine(source="torch", target="jax")
  code = "x = 1"
  tree = engine.parse(code)
  assert isinstance(tree, cst.Module)
  assert engine.to_source(tree) == code


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=True)
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline(mock_get_backend, mock_is_isa):
  """Verifies that the ASTEngine compiler pipeline executes successfully for SASS.

  This test ensures SASS-based assembly sources are successfully routed
  through the compiler pipeline (parser, lifter, and target backend compiler)
  and produce expected compiled output.

  Args:
      mock_get_backend (MagicMock): A mock representing the backend class retrieval.
      mock_is_isa (MagicMock): A mock assertion that the source language is an ISA.

  Returns:
      None
  """
  mock_backend_cls = MagicMock()
  mock_backend_instance = MagicMock()
  mock_backend_instance.compile.return_value = "compiled_code"
  mock_backend_cls.return_value = mock_backend_instance
  mock_backend_cls.__name__ = "MockBackend"
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="sass", target="jax")

  with (
    patch("ml_switcheroo.core.compiler.frontends.sass.SassParser") as mock_parser,
    patch("ml_switcheroo.core.compiler.frontends.sass.SassLifter") as mock_lifter,
  ):
    mock_parser.return_value.parse.return_value.statements = []
    mock_lifter.return_value.lift.return_value = MagicMock()

    result = engine.run("sass code")
    assert result.success is True
    assert result.code == "compiled_code"


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=False)
@patch("ml_switcheroo.core.engine.is_isa_target", return_value=False)
@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.engine.RewriterPipeline")
@patch("ml_switcheroo.core.engine.StructuralLinter")
def test_run_rewriter_pipeline(mock_linter_cls, mock_pipeline_cls, mock_ingest, mock_isa_target, mock_isa_source):
  """Verifies the rewrite pipeline runs successfully for framework-to-framework translation.

  This test simulates ingestion of code, routing it through the rewrite
  pipeline, and ensuring that no structural linting errors prevent success.

  Args:
      mock_linter_cls (MagicMock): A mock representing the StructuralLinter class.
      mock_pipeline_cls (MagicMock): A mock representing the RewriterPipeline class.
      mock_ingest (MagicMock): A mock of the code ingestion function.
      mock_isa_target (MagicMock): A mock assertion that target language is not an ISA.
      mock_isa_source (MagicMock): A mock assertion that source language is not an ISA.

  Returns:
      None
  """
  mock_tree = cst.parse_module("x = 1")
  mock_ingest.return_value = mock_tree

  mock_pipeline_instance = MagicMock()
  mock_pipeline_instance.run.return_value = mock_tree
  mock_pipeline_cls.return_value = mock_pipeline_instance

  mock_linter_instance = MagicMock()
  mock_linter_instance.check.return_value = []
  mock_linter_cls.return_value = mock_linter_instance

  engine = ASTEngine(source="torch", target="jax", strict_mode=True)
  result = engine.run("x = 1")

  assert result.success is True
  assert result.code == "x = 1"


def test_run_exception_handling():
  """Verifies that ASTEngine handles internal exceptions during run gracefully.

  This test ensures that when the underlying pipelines throw exceptions, the
  main run entrypoint catches them, flags success as False, and logs
  appropriate error diagnostics.

  Args:
      None

  Returns:
      None
  """
  engine = ASTEngine(source="torch", target="jax")
  with patch.object(engine, "_run_rewriter_pipeline", side_effect=Exception("Test Error")):
    result = engine.run("x = 1")
    assert result.success is False
    assert "Critical Failure: Test Error" in result.errors[0]


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=False)
@patch("ml_switcheroo.core.engine.is_isa_target", return_value=False)
@patch("ml_switcheroo.core.engine.ingest_code")
@patch("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter")
def test_run_stablehlo_pipeline(mock_emitter_cls, mock_ingest, mock_isa_target, mock_isa_source):
  """Verifies ASTEngine's translation pipeline to StableHLO format.

  This test simulates the path where the target is StableHLO, verifying
  the integration of StableHloEmitter to compile LibCST modules to MLIR/StableHLO code.

  Args:
      mock_emitter_cls (MagicMock): A mock representing the StableHloEmitter class.
      mock_ingest (MagicMock): A mock of the code ingestion function.
      mock_isa_target (MagicMock): A mock assertion that target language is not an ISA.
      mock_isa_source (MagicMock): A mock assertion that source language is not an ISA.

  Returns:
      None
  """
  mock_tree = cst.parse_module("x = 1")
  mock_ingest.return_value = mock_tree

  mock_emitter_instance = MagicMock()
  mock_result_node = MagicMock()
  mock_result_node.to_text.return_value = "mlir_code"
  mock_emitter_instance.convert.return_value = mock_result_node
  mock_emitter_cls.return_value = mock_emitter_instance

  engine = ASTEngine(source="torch", target="stablehlo")
  result = engine.run("x = 1")

  assert result.success is True
  assert result.code == "mlir_code"


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=True)
@patch("ml_switcheroo.core.engine.get_backend_class")
def test_run_compiler_pipeline_rdna(mock_get_backend, mock_is_isa):
  """Verifies that the ASTEngine compiler pipeline executes successfully for RDNA.

  This test ensures RDNA assembly sources are successfully routed
  through the compiler pipeline (parser, lifter, and target backend compiler)
  and produce expected compiled output.

  Args:
      mock_get_backend (MagicMock): A mock representing the backend class retrieval.
      mock_is_isa (MagicMock): A mock assertion that the source language is an ISA.

  Returns:
      None
  """
  mock_backend_cls = MagicMock()
  mock_backend_instance = MagicMock()
  mock_backend_instance.compile.return_value = "compiled_rdna"
  mock_backend_cls.return_value = mock_backend_instance
  mock_backend_cls.__name__ = "MockBackend"
  mock_get_backend.return_value = mock_backend_cls

  engine = ASTEngine(source="rdna", target="jax")

  with (
    patch("ml_switcheroo.core.compiler.frontends.rdna.RdnaParser") as mock_parser,
    patch("ml_switcheroo.core.compiler.frontends.rdna.RdnaLifter") as mock_lifter,
  ):
    mock_parser.return_value.parse.return_value.statements = []
    mock_lifter.return_value.lift.return_value = MagicMock()

    result = engine.run("rdna code")
    assert result.success is True
    assert result.code == "compiled_rdna"


@patch("ml_switcheroo.core.engine.is_isa_source", return_value=True)
def test_run_compiler_pipeline_unsupported_frontend(mock_is_isa):
  """Verifies ASTEngine behavior when requesting an unsupported ISA source.

  This test asserts that when an ISA source language is requested but no valid
  parser/lifter frontend is available, the engine reports failure gracefully.

  Args:
      mock_is_isa (MagicMock): A mock assertion that the source language is an ISA.

  Returns:
      None
  """
  engine = ASTEngine(source="html", target="jax")
  result = engine.run("code")
  assert result.success is False
  assert "No frontend for html" in result.errors[0]


def test_astengine_graph_to_mermaid():
  """Verifies ASTEngine can convert a LibCST module tree to a Mermaid visualization.

  This test asserts that ASTEngine produces a string representation formatted
  as Mermaid graph syntax representing the structure of parsed code.

  Args:
      None

  Returns:
      None
  """
  engine = ASTEngine(source="torch", target="jax")
  tree = cst.parse_module("x = 1")
  mermaid = engine._graph_to_mermaid(tree)
  assert isinstance(mermaid, str)


def test_astengine_strict_mode():
  """Verifies ASTEngine handles the strict_mode initialization attribute.

  This test asserts that ASTEngine stores and accesses the strict_mode flag correctly
  with default values when not explicitly overridden.

  Args:
      None

  Returns:
      None
  """
  engine = ASTEngine(source="torch", target="jax")
  assert getattr(engine, "strict_mode") is False
