"""
Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, generating code via:
1.  **Compiler Pipeline**: For ISA/Visuals (Source -> Graph -> Backend -> Target).
2.  **Rewriter Pipeline**: For High-Level Frameworks (Source -> CST -> Pipeline(Structure, API, Aux) -> Target).
    - Supports optional **Graph-Guided Rewriting** (Loopback).
"""

from typing import Any, Dict, Optional, cast
import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.core.tracer import get_tracer, reset_tracer
from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.graph import GraphExtractor

# Rewriter Components
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass
from ml_switcheroo.core.rewriter.passes.api import ApiPass
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass

# Compiler Components
from ml_switcheroo.compiler.registry import (
  is_isa_target,
  is_isa_source,
  get_backend_class,
)
from ml_switcheroo.compiler.frontends.python import PythonFrontend
from ml_switcheroo.compiler.frontends.sass import SassParser, SassLifter
from ml_switcheroo.compiler.frontends.rdna import RdnaParser, RdnaLifter
from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.frameworks.base import get_adapter

# Visualization
from ml_switcheroo.utils.visualizer import MermaidGenerator


class ASTEngine:
  """The main driver for the conversion process."""

  def __init__(
    self,
    semantics: Optional[SemanticsManager] = None,
    config: Optional[RuntimeConfig] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    strict_mode: bool = False,
    enable_graph_optimization: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
    intermediate: Optional[str] = None,
  ) -> None:
    """
    Initializes the engine with semantics and configuration.

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
    """Main execution point."""
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Pipeline Start", f"{self.source} -> {self.target}")

    try:
      if is_isa_source(self.source) or is_isa_target(self.target):
        result = self._run_compiler_pipeline(code, tracer)
      else:
        result = self._run_rewriter_pipeline(code, tracer)
      tracer.end_phase()
      return result
    except Exception as e:
      tracer.end_phase()
      return ConversionResult(
        code=code,
        errors=[f"Critical Failure: {str(e)}"],
        success=False,
        trace_events=tracer.export(),
      )

  def parse(self, code: str) -> cst.Module:
    """Parses python string to CST Module."""
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Converts CST Module back to string."""
    return tree.code

  def _graph_to_mermaid(self, tree: cst.CSTNode) -> str:
    """Helper to generate mermaid definition from CST."""
    return MermaidGenerator().generate(tree)

  def _run_compiler_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    """Runs the Graph-based compiler pipeline for ISAs."""
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
      except Exception:
        # Fallback: Maybe it's valid python code already
        frontend = PythonFrontend(code)
        graph = frontend.parse_to_graph()

    else:
      # ISA Source (SASS/RDNA) logic
      if self.source == "sass":
        parser = SassParser(code)
        nodes = parser.parse()
        lifter = SassLifter()
        graph = lifter.lift(nodes)
      elif self.source == "rdna":
        parser = RdnaParser(code)
        nodes = parser.parse()
        lifter = RdnaLifter()
        graph = lifter.lift(nodes)
      else:
        raise NotImplementedError(f"No frontend for {self.source}")

    if self.config.enable_graph_optimization:
      from ml_switcheroo.core.graph_optimizer import GraphOptimizer

      tracer.start_phase("Optimization", "Fusion")
      patterns = self.semantics.get_patterns()
      optimizer = GraphOptimizer(patterns)
      graph = optimizer.optimize(graph)
      tracer.log_mutation("Graph Optimization", "(Graph)", "(Optimized Graph)")
      tracer.log_snapshot("After Optimization", "graph TD; A-->B;", "...")
      tracer.end_phase()

    backend_cls = get_backend_class(self.target)
    if not backend_cls:
      raise ValueError(f"No backend found for {self.target}")

    if backend_cls.__name__ == "PythonBackend":
      backend = backend_cls(framework=self.target)  # type: ignore
    else:
      backend = cast(CompilerBackend, backend_cls(self.semantics))

    output_code = backend.compile(graph)
    tracer.log_mutation("Codegen", "(Graph)", output_code)
    tracer.end_phase()
    return ConversionResult(code=output_code, success=True, trace_events=tracer.export())

  def _run_rewriter_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    """Structural pipeline with optional graph loopback."""
    tracer.start_phase("Rewriter Pipeline", "AST Transformation")

    # 1. Ingestion
    source_adapter = get_adapter(self.source)
    tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
    tracer.log_snapshot("After Ingestion", self._graph_to_mermaid(tree), self.to_source(tree))

    # 1.5. Graph-Guided Optimization (The "Loopback")
    if self.config.enable_graph_optimization:
      tracer.start_phase("Graph Guided Rewriting", "Fusion & Surgery")
      try:
        from ml_switcheroo.core.graph_optimizer import GraphOptimizer
        from ml_switcheroo.compiler.differ import GraphDiffer
        from ml_switcheroo.core.rewriter.patcher import GraphPatcher
        from ml_switcheroo.compiler.backends.python_snippet import (
          PythonSnippetEmitter,
        )

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
      except Exception as e:
        tracer.log_warning(f"Graph Optimization failed, proceeding with raw CST: {e}")
      tracer.end_phase()

    # 2. Analysis
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
        ApiPass(),  # Core logic, calls, attributes
        AuxiliaryPass(),  # Decorators and safety mechanisms
      ]
    )

    tree = pipeline.run(tree, context)
    tracer.log_snapshot("After Rewriting", self._graph_to_mermaid(tree), self.to_source(tree))

    # 4. Import Fixing
    if self.config.enable_import_fixer:
      usage_scanner = UsageScanner(self.source)
      tree.visit(usage_scanner)
      should_preserve = usage_scanner.get_result()
      resolver = ImportResolver(self.semantics)
      plan = resolver.resolve(tree, self.target)
      fixer = ImportFixer(
        plan=plan,
        source_fws={self.source},
        preserve_source=should_preserve,
      )
      tree = tree.visit(fixer)
      tracer.log_snapshot("After Import Fixing", self._graph_to_mermaid(tree), self.to_source(tree))

    # 5. Emission
    final_code = tree.code

    # Trace
    if self.target == "mlir" or self.target == "stablehlo":
      tracer.log_mutation("Final Emission", "(Python CST)", final_code)

    # 6. Checks
    errors = []
    if EscapeHatch.START_MARKER in final_code:
      msg = "Escape Hatches Detected: Partial conversion. Inspect output for '# <SWITCHEROO...' blocks."
      errors.append(msg)
      tracer.log_warning(msg)

    if self.strict_mode and self.target not in ["mlir", "stablehlo", "latex_dsl", "tikz"]:
      tracer.start_phase("Structural Linter", "Safety Verification")
      linter = StructuralLinter(forbidden_roots={self.source})
      list_errors = linter.check(final_code)
      if list_errors:
        tracer.log_warning(f"Linter errors: {list_errors}")
        errors.extend(list_errors)
      tracer.end_phase()

    tracer.end_phase()
    return ConversionResult(
      code=final_code,
      success=True,
      errors=errors,
      trace_events=tracer.export(),
    )

  @property
  def strict_mode(self) -> bool:
    """Helper property for strict mode config."""
    return self.config.strict_mode
