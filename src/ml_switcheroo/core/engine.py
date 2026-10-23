"""Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, generating code via:

 **Compiler Pipeline**: For ISA/Visuals (Source -> Graph -> Backend -> Target).

**Rewriter Pipeline**: For High-Level Frameworks (Source -> CST -> Pipeline(Structure, API, Aux) -> Target).

Supports optional **Graph-Guided Rewriting** (Loopback).
"""

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser

# Compiler Components
from ml_switcheroo.core.compiler.registry import (
  _FRONTENDS,
  get_backend_class,
  instantiate_backend,
  is_isa_source,
)
from ml_switcheroo.core.conversion_result import ConversionResult as ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.core.ingestion import ingest_code

# Rewriter Components
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiPass
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass
from ml_switcheroo.core.rewriter.passes.mutation import FunctionalMutationPass
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.scanners import GlobalUsageScanner
from ml_switcheroo.core.tracer import TraceLogger, get_tracer, reset_tracer
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.linter import StructuralLinter

# Visualization
from ml_switcheroo.utils.visualizer import MermaidGenerator


class ASTEngine:
  """Execute main driver for the conversion process."""

  def __init__(
    self,
    semantics: SemanticsManager | None = None,
    config: RuntimeConfig | None = None,
    source: str | None = None,
    target: str | None = None,
    strict_mode: bool = False,
    enable_graph_optimization: bool = False,
    plugin_config: dict[str, JsonValue] | None = None,
    intermediate: str | None = None,
  ) -> None:
    """Initialize the engine with semantics and configuration.

    Args:
        semantics: Valid SemanticsManager instance.
        config: Runtime configuration object.
        source: Source framework key override.
        target: Target framework key override.
        strict_mode: Whether to fail on unmapped operations.
        enable_graph_optimization: Whether to run fusion pass.
        plugin_config: Dictionary of settings for plugins.
        intermediate: Intermediate generation format.

    """
    self.semantics: SemanticsManager = semantics or SemanticsManager()
    if config:
      self.config = config
      if intermediate is not None:
        self.config.intermediate = intermediate
    else:
      self.config = RuntimeConfig.load(
        source=source,
        target=target,
        strict_mode=strict_mode,
        intermediate=intermediate,
        enable_graph_optimization=enable_graph_optimization,
        plugin_settings=plugin_config or {},
      )
    if self.config.validation_report:
      self.semantics.load_validation_report(self.config.validation_report)

    self.source = self.config.effective_source
    self.target = self.config.effective_target
    load_plugins()

  def run(self, code: str) -> ConversionResult:
    """Execute the complete conversion pipeline on the input code.

    Args:
        code: Input source code string to convert.

    Returns:
        ConversionResult containing the transpiled code, diagnostics, and trace events.
    """
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Pipeline Start", f"{self.source} -> {self.target}")

    try:
      if self.config.intermediate in ["ir", "ml_switcheroo_ir"]:
        tracer.start_phase("Intermediate Pipeline", f"{self.source} -> IR -> {self.target}")
        engine_to_ir = ASTEngine(
          semantics=self.semantics,
          source=self.source,
          target="ir",
          strict_mode=self.strict_mode,
        )
        ir_result = engine_to_ir.run(code)
        if not ir_result.success:
          tracer.end_phase()
          return ir_result

        engine_from_ir = ASTEngine(
          semantics=self.semantics,
          source="ir",
          target=self.target,
          strict_mode=self.strict_mode,
        )
        final_result = engine_from_ir.run(ir_result.code)
        tracer.end_phase()
        return final_result

      from ml_switcheroo.core.strategies import STRATEGY_REGISTRY

      for strategy in STRATEGY_REGISTRY:
        if strategy.can_handle(self):
          result = strategy.execute(self, code, tracer)
          break

      tracer.end_phase()
      return result
    except Exception as e:
      import logging

      logging.error(f"ASTEngine Error: {e}", exc_info=True)
      tracer.end_phase()
      return ConversionResult(
        code=code,
        errors=[f"Critical Failure: {str(e)}"],
        success=False,
        trace_events=tracer.export(),
      )

  def parse(self, code: str) -> cst.Module:
    """Parse a Python source code string into a LibCST Module.

    Args:
        code: The Python source code as a string.

    Returns:
        A libcst.Module representation of the input source code.
    """
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Convert a LibCST Module back into its source code string representation.

    Args:
        tree: The LibCST Module to be serialized.

    Returns:
        The generated source code as a string.
    """
    return tree.code

  def _graph_to_mermaid(self, tree: cst.CSTNode) -> str:
    """Generate a Mermaid diagram definition representing the CST structure.

    Args:
        tree: The LibCST node to visualize.

    Returns:
        A string containing the Mermaid diagram definition.
    """
    return MermaidGenerator().generate(tree)

  def run_compiler_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
    """Execute the Graph-based compiler pipeline.

    Args:
        code: The input source code string.
        tracer: The execution tracer to log phases and mutations.

    Returns:
        A ConversionResult containing the compiled target code and execution trace.
    """
    return self._run_compiler_pipeline(code, tracer)

  def _run_compiler_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
    """Run the Graph-based compiler pipeline for Instruction Set Architectures (ISAs).

    This pipeline handles conversion for targets like NVIDIA_SASS, RDNA, or graph-level
    optimizations such as sharding. It parses the source into a graph, optimizes it
    (if enabled), applies sharding inference, and then runs a compiler backend
    to emit target code.

    Args:
        code: The input source code string.
        tracer: The execution tracer to log phases and mutations.

    Returns:
        A ConversionResult containing the compiled target code and execution trace.

    Raises:
        NotImplementedError: If no frontend is available for the source framework.
        ValueError: If no backend is found for the target framework.
    """
    tracer.start_phase("Compiler Pipeline", f"{self.source}->Graph->{self.target}")
    graph = None
    if not is_isa_source(self.source):
      # Special case: MLIR source ingestion is handled by ingest_code which returns CST,
      # but for MLIR roundtrip we want to reach MlirBackend via graph only if possible.
      # But MlirToPythonGenerator produces CST.
      # If source is MLIR and target is MLIR, logic requires special bridge.
      # The Registry identifies 'python' as the frontend for 'mlir', which is PythonFrontend.
      # PythonFrontend works on Python Code (CST).
      # If we pass MLIR *text* to PythonFrontend, it will fail to parse as python.
      # We need to use ingest_code's logic to parse MLIR to Python CST first IF we need python intermediate.

      # However, `is_isa_source` returns False for MLIR. So we arrive here.
      # If source is not an ISA, assume python structure.
      # If the input text is actually MLIR, PythonFrontend crashes.
      # We must use ingest_code to normalize to AST, THEN extract Graph.
      try:
        # We reuse the ingestion logic to get a Python CST from the source, whatever it is
        source_adapter = get_adapter(self.source)
        cst_tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
        code_for_graph = self.to_source(cst_tree)
        frontend = PythonFrontend(code_for_graph)
        graph = frontend.parse_to_graph()
      except (cst.ParserSyntaxError, SyntaxError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as exc:
        import logging

        logging.getLogger(__name__).debug("Ingestion fallback to raw PythonFrontend: %s", exc)
        tracer.log_warning(f"Ingestion failed ({exc}); falling back to direct PythonFrontend parsing.")
        frontend = PythonFrontend(code)
        graph = frontend.parse_to_graph()

    else:
      # ISA / Graph Source logic via Compiler Registry _FRONTENDS
      isa_overrides = {
        "nvidia_sass": (NvidiaSassParser, NvidiaSassLifter),
        "rdna": (RdnaParser, RdnaLifter),
      }
      frontend_entry = isa_overrides.get(self.source) or _FRONTENDS.get(self.source)
      if isinstance(frontend_entry, tuple) and len(frontend_entry) == 2:
        parser_cls, lifter_cls = frontend_entry
        isa_parser = parser_cls(code)
        isa_nodes = isa_parser.parse().statements
        isa_lifter = lifter_cls()
        graph = isa_lifter.lift(isa_nodes)
      elif callable(frontend_entry) and hasattr(frontend_entry, "parse_to_graph"):
        graph_frontend = frontend_entry(code)
        parse_fn = getattr(graph_frontend, "parse_to_graph")
        graph = parse_fn()
      elif frontend_entry is not None:
        source_adapter = get_adapter(self.source)
        cst_tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
        code_for_graph = self.to_source(cst_tree)
        frontend = PythonFrontend(code_for_graph)
        graph = frontend.parse_to_graph()
      else:
        raise NotImplementedError(f"No frontend for {self.source}")

    assert graph is not None

    if self.config.enable_graph_optimization:
      from ml_switcheroo.core.graph_optimizer import GraphOptimizer

      tracer.start_phase("Optimization", "Fusion")
      patterns = self.semantics.get_patterns()
      optimizer = GraphOptimizer(patterns)
      graph = optimizer.optimize(graph)
      tracer.log_mutation("Graph Optimization", "(Graph)", "(Optimized Graph)")
      tracer.log_snapshot("After Optimization", "graph TD; A-->B;", "...")
      tracer.end_phase()

    if getattr(self.config, "enable_sharding", False):
      from ml_switcheroo.core.compiler.sharding import apply_sharding_passes

      graph = apply_sharding_passes(graph, fuse=self._is_sharding_fusion_target())

    backend_cls = get_backend_class(self.target)
    if not backend_cls:
      raise ValueError(f"No backend found for {self.target}")

    backend: CompilerBackend = instantiate_backend(
      backend_cls=backend_cls,
      target=self.target,
      semantics=self.semantics,
    )

    warnings: list[str] = []
    try:
      from ml_switcheroo.analysis.static_safety import StaticSafetyScanner

      safety_diags = StaticSafetyScanner.scan(code)
      for diag in safety_diags:
        warning_msg = f"[{diag.category.value}] {diag.message} Recommendation: {diag.recommendation}"
        warnings.append(warning_msg)
        tracer.log_warning(warning_msg)
    except (cst.ParserSyntaxError, SyntaxError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
      import logging

      logging.getLogger(__name__).debug("StaticSafetyScanner skipped in compiler pipeline: %s", exc)

    output_code = backend.compile(graph)
    tracer.log_mutation("Codegen", "(Graph)", output_code)
    tracer.end_phase()
    return ConversionResult(
      code=output_code,
      success=True,
      warnings=warnings,
      trace_events=tracer.export(),
    )

  def run_rewriter_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
    """Execute the structural CST rewriter pipeline.

    Args:
        code: The input source code string.
        tracer: The execution tracer to log phases and mutations.

    Returns:
        A ConversionResult containing the rewritten code and diagnostics.
    """
    return self._run_rewriter_pipeline(code, tracer)

  def _run_rewriter_pipeline(self, code: str, tracer: TraceLogger) -> ConversionResult:
    """Run the structural rewriter pipeline with optional graph loopback optimization.

    This pipeline performs AST-to-AST rewriting for high-level frameworks. It
    first ingests the code into a CST, optionally extracts and optimizes the
    computation graph (loopback), patches the CST with the optimized graph, and
    then applies structural, API, and auxiliary passes followed by import fixing.

    Args:
        code: The input source code string.
        tracer: The execution tracer to log phases and mutations.

    Returns:
        A ConversionResult containing the rewritten code, status, and any
        warnings or validation errors.
    """
    tracer.start_phase("Rewriter Pipeline", "AST Transformation")

    # 1. Ingestion
    source_adapter = get_adapter(self.source)
    tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
    tracer.log_snapshot("After Ingestion", self._graph_to_mermaid(tree), self.to_source(tree))

    # 1.5. Graph-Guided Optimization (The "Loopback")
    if self.config.enable_graph_optimization:
      tracer.start_phase("Graph Guided Rewriting", "Fusion & Surgery")
      try:
        from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
        from ml_switcheroo.core.compiler.differ import GraphDiffer
        from ml_switcheroo.core.graph_optimizer import GraphOptimizer
        from ml_switcheroo.core.rewriter.patcher import GraphPatcher

        # A. Extraction
        extractor = GraphExtractor()
        tree.visit(extractor)
        original_graph = extractor.graph
        provenance = extractor.node_map

        if original_graph.nodes:
          # B. Optimization
          patterns = self.semantics.get_patterns()
          optimizer = GraphOptimizer(patterns)
          optimized_graph = optimizer.optimize(original_graph)

          if self.config.enable_sharding:
            from ml_switcheroo.core.compiler.sharding import apply_sharding_passes

            optimized_graph = apply_sharding_passes(optimized_graph, fuse=self._is_sharding_fusion_target())

          # C. Differ
          differ = GraphDiffer()
          plan = differ.diff(original_graph, optimized_graph)

          # D. Patching
          if plan:
            emitter = PythonSnippetEmitter(framework=self.target)
            patcher = GraphPatcher(plan, provenance, emitter)
            tree = tree.visit(patcher)
            tracer.log_mutation(
              "Graph Patching",
              "Original CST",
              self.to_source(tree),
            )
            tracer.log_snapshot(
              "After Graph Patching",
              self._graph_to_mermaid(tree),
              self.to_source(tree),
            )
          else:
            tracer.log_mutation("Graph Patching", "No Plan", "Skipped")
      except (cst.ParserSyntaxError, SyntaxError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
        tracer.log_warning(f"Graph Optimization failed, proceeding with raw CST: {e}")

      tracer.end_phase()

    # - Analysis
    tracer.log_snapshot("After Analysis", self._graph_to_mermaid(tree), self.to_source(tree))

    # 3. Rewriting (Pipeline)
    # Construct Context
    context = RewriterContext(
      semantics=self.semantics,
      config=self.config,
      # Symbol table logic can be injected here if analysis passes are added to engine
      symbol_table=None,
    )

    # Construct Pipeline
    pipeline = RewriterPipeline(
      [
        StructuralPass(),  # Class and signature changes
        FunctionalMutationPass(),  # In-place mutations, indexing, PRNG state
        ApiPass(),  # Core logic, calls, attributes
        AuxiliaryPass(),  # Decorators and safety mechanisms
      ]
    )

    tree = pipeline.run(tree, context)
    tracer.log_snapshot("After Rewriting", self._graph_to_mermaid(tree), self.to_source(tree))

    # 4. Import Fixing
    if self.config.enable_import_fixer:
      usage_scanner = GlobalUsageScanner()
      tree.visit(usage_scanner)
      used_names = usage_scanner.used_names

      resolver = ImportResolver(self.semantics)
      resolution_plan = resolver.resolve(tree, self.target)

      source_frameworks = self.semantics.get_framework_ecosystem(self.config.source_framework)
      source_frameworks.update(self.semantics.get_framework_ecosystem(self.config.effective_source))

      fixer = ImportFixer(
        plan=resolution_plan,
        source_frameworks=source_frameworks,
        used_names=used_names,
      )
      tree = tree.visit(fixer)
      tracer.log_snapshot("After Import Fixing", self._graph_to_mermaid(tree), self.to_source(tree))

    # 5. Emission
    final_code = tree.code

    # Trace

    # 6. Checks
    errors = []
    if EscapeHatch.START_MARKER in final_code:
      msg = "Escape Hatches Detected: Partial conversion. Inspect output for '# <SWITCHEROO...' blocks."
      errors.append(msg)
      tracer.log_warning(msg)

    if self.strict_mode:
      tracer.start_phase("Structural Linter", "Safety Verification")
      linter = StructuralLinter(forbidden_roots={self.source})
      list_errors = linter.check(final_code)
      if list_errors:
        tracer.log_warning(f"Linter errors: {list_errors}")
        errors.extend(list_errors)
      tracer.end_phase()

    warnings: list[str] = []
    try:
      from ml_switcheroo.analysis.static_safety import StaticSafetyScanner

      safety_diags = StaticSafetyScanner.scan(tree)
      for diag in safety_diags:
        warning_msg = f"[{diag.category.value}] {diag.message} Recommendation: {diag.recommendation}"
        warnings.append(warning_msg)
        tracer.log_warning(warning_msg)
    except (cst.ParserSyntaxError, SyntaxError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
      import logging

      logging.getLogger(__name__).debug("StaticSafetyScanner skipped in rewriter pipeline: %s", exc)

    tracer.end_phase()
    return ConversionResult(
      code=final_code,
      success=True,
      errors=errors,
      warnings=warnings,
      trace_events=tracer.export(),
    )

  def _is_sharding_fusion_target(self) -> bool:
    """Check whether the target framework requires sharded QKV/SwiGLU/VisionPatch fusion passes.

    Returns:
        True if target framework uses functional XLA/pjit sharding fusion.
    """
    adapter = get_adapter(self.target)
    if adapter is not None:
      return bool(adapter.plugin_traits.supports_sharding or adapter.plugin_traits.enforce_purity_analysis)
    return False

  @property
  def strict_mode(self) -> bool:
    """Support property to retrieve the strict mode setting from config.

    Returns:
        True if strict mode is enabled, False otherwise.
    """
    return self.config.strict_mode
