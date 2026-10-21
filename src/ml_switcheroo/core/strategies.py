"""Pipeline Strategies for ASTEngine.

Implements the Strategy pattern to decouple different compilation and translation
pipelines (e.g., Rewriter, Compiler, Visualization).
"""

import typing

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.tracer import TraceLogger

  class EngineHost(typing.Protocol):
    """Typed protocol for the orchestration engine executing pipeline strategies."""

    source: str
    target: str
    config: RuntimeConfig
    semantics: SemanticsManager

    def run_compiler_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
      """Execute the graph-based compiler pipeline.

      Args:
          code: Input source code string.
          tracer: Execution tracer instance.

      Returns:
          ConversionResult containing compiled code and diagnostics.
      """
      ...

    def run_rewriter_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
      """Execute the CST-based high-level rewriter pipeline.

      Args:
          code: Input source code string.
          tracer: Execution tracer instance.

      Returns:
          ConversionResult containing rewritten code and diagnostics.
      """
      ...


class PipelineStrategy:
  """Base class for an execution strategy in the ASTEngine."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Determine if this strategy should handle the current engine configuration.

    Args:
        engine: The orchestration engine instance holding source/target config.

    Returns:
        True if this strategy supports the configured conversion, False otherwise.
    """
    raise NotImplementedError()

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Execute the conversion logic for the given source code.

    Args:
        engine: The orchestration engine instance.
        code: The input source code string to convert.
        tracer: The execution tracer capturing pipeline phases and mutations.

    Returns:
        A ConversionResult with the generated code, status, and trace events.
    """
    raise NotImplementedError()


class VisualStrategy(PipelineStrategy):
  """Strategy for HTML, TikZ, and LaTeX visualizations."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Check whether the target is a visual markup DSL (HTML, TikZ, or LaTeX).

    Args:
        engine: The orchestration engine instance.

    Returns:
        True if the target framework is html, tikz, or latex_dsl.
    """
    return engine.target in ["html", "tikz", "latex_dsl"]

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Execute the compiler graph pipeline to render visual markup.

    Args:
        engine: The orchestration engine instance.
        code: The input source code string.
        tracer: The execution tracer instance.

    Returns:
        A ConversionResult containing rendered visual markup.
    """
    return engine.run_compiler_pipeline(code, tracer)


class MlirCompilationStrategy(PipelineStrategy):
  """Strategy for MLIR and StableHLO targets."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Check whether the target is MLIR or StableHLO.

    Args:
        engine: The orchestration engine instance.

    Returns:
        True if the target is mlir or stablehlo.
    """
    return engine.target in ["mlir", "stablehlo"]

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Execute MLIR or StableHLO emission from the ingested CST or graph.

    Args:
        engine: The orchestration engine instance.
        code: The input source code string.
        tracer: The execution tracer instance.

    Returns:
        A ConversionResult containing emitted MLIR or StableHLO text.
    """
    from ml_switcheroo.core.ingestion import ingest_code
    from ml_switcheroo.frameworks.base import get_adapter

    source_adapter = get_adapter(engine.source)
    tree = ingest_code(code, engine.source, engine.target, source_adapter, tracer)

    if engine.target == "stablehlo":
      from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter

      emitter = StableHloEmitter(engine.semantics)
      mlir_code = emitter.convert(tree).to_text()
      return ConversionResult(code=mlir_code, success=True, trace_events=tracer.export())
    return engine.run_compiler_pipeline(code, tracer)


class IsaCompilationStrategy(PipelineStrategy):
  """Strategy for hardware assembly targets and sources (SASS, RDNA) or sharded graphs."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Check whether the conversion involves an ISA source/target or sharding.

    Args:
        engine: The orchestration engine instance.

    Returns:
        True if the source or target is an ISA or if sharding is enabled.
    """
    from ml_switcheroo.core.compiler.registry import is_isa_source, is_isa_target

    return is_isa_source(engine.source) or is_isa_target(engine.target) or engine.config.enable_sharding

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Execute the compiler graph pipeline for ISA lifting/synthesis or sharding.

    Args:
        engine: The orchestration engine instance.
        code: The input source code string.
        tracer: The execution tracer instance.

    Returns:
        A ConversionResult containing the compiled target code.
    """
    return engine.run_compiler_pipeline(code, tracer)


class CstRewriterStrategy(PipelineStrategy):
  """Fallback strategy for high-level Python framework CST transpilation."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Always returns True as the default fallback strategy.

    Args:
        engine: The orchestration engine instance.

    Returns:
        True unconditionally.
    """
    return True

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Execute the high-level CST rewriter pipeline.

    Args:
        engine: The orchestration engine instance.
        code: The input source code string.
        tracer: The execution tracer instance.

    Returns:
        A ConversionResult containing the rewritten Python code.
    """
    return engine.run_rewriter_pipeline(code, tracer)


# Registry of strategies, order matters (Fallback should be last)
STRATEGY_REGISTRY: list[PipelineStrategy] = [
  VisualStrategy(),
  MlirCompilationStrategy(),
  IsaCompilationStrategy(),
  CstRewriterStrategy(),
]
