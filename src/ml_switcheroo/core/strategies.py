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
    """Typed protocol for the orchestration engine executing pipeline strategies.

    Attributes:
        source: The source (`str`).
        target: The target (`str`).
        config: The config (`RuntimeConfig`).
        semantics: The semantics (`SemanticsManager`).
    """

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

    Raises:
        NotImplementedError: Must be implemented by concrete strategy subclasses.
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

    Raises:
        NotImplementedError: Must be implemented by concrete strategy subclasses.
    """
    raise NotImplementedError()


class IntermediateMlirStrategy(PipelineStrategy):
  """Strategy for ``intermediate='mlir'`` and ``intermediate='stablehlo'`` multi-hop roundtrips."""

  def can_handle(self, engine: "EngineHost") -> bool:
    """Check whether the engine requests an MLIR or StableHLO intermediate bridge.

    Args:
        engine: The orchestration engine instance.

    Returns:
        True when ``intermediate`` is ``'mlir'`` or ``'stablehlo'`` between non-MLIR endpoints.
    """
    return (
      engine.config.intermediate in ("mlir", "stablehlo")
      and engine.source not in ("mlir", "stablehlo")
      and engine.target not in ("mlir", "stablehlo")
    )

  def execute(self, engine: "EngineHost", code: str, tracer: "TraceLogger") -> ConversionResult:
    """Lower ``code`` to MLIR or StableHLO, parse/lift back, and emit target framework code.

    Args:
        engine: The orchestration engine instance.
        code: Input source code string.
        tracer: Scoped execution tracer instance.

    Returns:
        A ConversionResult produced after roundtripping through MLIR or StableHLO.
    """
    from ml_switcheroo.core.engine import ASTEngine, EngineExecutionContext
    from ml_switcheroo.core.ingestion import ingest_code
    from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
    from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
    from ml_switcheroo.core.mlir.parser import MlirParser
    from ml_switcheroo.frameworks.base import get_adapter

    exec_ctx = EngineExecutionContext(tracer=tracer)

    if engine.config.intermediate == "stablehlo":
      tracer.start_phase("StableHLO Bridge", f"{engine.source} -> StableHLO -> {engine.target}")
      to_shlo_cfg = engine.config.model_copy(
        update={
          "target_framework": "stablehlo",
          "target_flavour": None,
          "intermediate": None,
        }
      )
      to_shlo_res = ASTEngine(semantics=engine.semantics, config=to_shlo_cfg).run(
        code,
        execution_context=exec_ctx,
      )
      if not to_shlo_res.success:
        tracer.end_phase()
        return to_shlo_res

      from_shlo_cfg = engine.config.model_copy(
        update={
          "source_framework": "stablehlo",
          "source_flavour": None,
          "intermediate": None,
        }
      )
      final_shlo_res = ASTEngine(semantics=engine.semantics, config=from_shlo_cfg).run(
        to_shlo_res.code,
        execution_context=exec_ctx,
      )
      tracer.end_phase()
      return final_shlo_res

    tracer.start_phase("MLIR Bridge", f"{engine.source} -> MLIR -> {engine.target}")
    source_adapter = get_adapter(engine.source)
    source_tree = ingest_code(code, engine.source, "mlir", source_adapter, tracer)
    mlir_module = PythonToMlirEmitter().convert(source_tree)
    mlir_text = mlir_module.to_text()
    tracer.log_mutation("MLIR Lowering", code, mlir_text)

    parsed_mlir = MlirParser(mlir_text).parse()
    lifted_tree = MlirToPythonGenerator().generate(parsed_mlir)
    lifted_code = lifted_tree.code
    tracer.log_mutation("MLIR Lifting", mlir_text, lifted_code)

    direct_cfg = engine.config.model_copy(update={"intermediate": None})
    final_mlir_res = ASTEngine(semantics=engine.semantics, config=direct_cfg).run(
      lifted_code,
      execution_context=exec_ctx,
    )
    tracer.end_phase()
    return final_mlir_res


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
    from ml_switcheroo.core.compiler.registry import is_isa_source
    from ml_switcheroo.core.ingestion import ingest_code
    from ml_switcheroo.core.rewriter.passes.runtime_annotations import SourceLazyConcretizer
    from ml_switcheroo.core.runtime_models import RuntimeTraceReport
    from ml_switcheroo.core.runtime_tracer import run_on_device_trace
    from ml_switcheroo.frameworks.base import get_adapter

    if is_isa_source(engine.source):
      return engine.run_compiler_pipeline(code, tracer)

    source_adapter = get_adapter(engine.source)
    tree = ingest_code(code, engine.source, engine.target, source_adapter, tracer)

    if engine.target == "stablehlo":
      from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter

      runtime_trace: RuntimeTraceReport | None = None
      if engine.config.enable_on_device_execution:
        from ml_switcheroo.core.rewriter.context import RewriterContext
        from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass

        tracer.start_phase("Runtime Forward & Backward Pass", f"On-Device ({engine.source} -> {engine.target})")
        runtime_trace = run_on_device_trace(
          code=code,
          source_fw=engine.source,
          target_fw=engine.target,
          semantics=engine.semantics,
          config=engine.config,
        )
        tree = tree.visit(SourceLazyConcretizer(runtime_trace))
        aux_ctx = RewriterContext(
          semantics=engine.semantics,
          config=engine.config,
          runtime_trace=runtime_trace,
        )
        tree = AuxiliaryPass().transform(tree, aux_ctx)
        tracer.end_phase()

      emitter = StableHloEmitter(engine.semantics, runtime_trace=runtime_trace)
      mlir_code = emitter.convert(tree).to_text()
      return ConversionResult(
        code=mlir_code,
        success=True,
        trace_events=tracer.export(),
        runtime_trace=runtime_trace,
      )
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
        engine: The host ASTEngine instance.
        code: Raw source code string.
        tracer: Execution trace logger.

    Returns:
        A ConversionResult containing the compiled target code.
    """
    from ml_switcheroo.core.compiler.registry import _FRONTENDS, is_isa_target
    from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser

    if (
      engine.config.enable_on_device_execution
      and not is_isa_target(engine.target)
      and _FRONTENDS.get(engine.source) is StableHloParser
    ):
      return engine.run_rewriter_pipeline(code, tracer)
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
  IntermediateMlirStrategy(),
  VisualStrategy(),
  MlirCompilationStrategy(),
  IsaCompilationStrategy(),
  CstRewriterStrategy(),
]
