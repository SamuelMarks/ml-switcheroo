"""
Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, the primary driver for the transpilation process.
It coordinates the various analysis and transformation passes.

Updates:
- Integrates Snapshot emission at phase boundaries.
- Supports conditional Graph Optimization and Import Fixing logic via RuntimeConfig.
- Captures source code alongside AST graphs for "Time Travel" debugging.
"""

from typing import Any, Dict, Optional, List

import libcst as cst

import ml_switcheroo.frameworks as fw_registry
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.analysis.lifecycle import InitializationTracker
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir_bridge import run_mlir_roundtrip
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.utils.visualizer import MermaidGenerator
from ml_switcheroo.core.tracer import get_tracer, reset_tracer


class ASTEngine:
  """
  The main compilation unit.

  Orchestrates parsing, analysis, transformation, and code generation.
  """

  def __init__(
    self,
    semantics: Optional[SemanticsManager] = None,
    config: Optional[RuntimeConfig] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    strict_mode: bool = False,
    enable_fusion: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
    intermediate: Optional[str] = None,
  ):
    """
    Initializes the Engine.
    """
    self.semantics = semantics or SemanticsManager()

    if config:
      self.config = config
    else:
      self.config = RuntimeConfig.load(
        source=source,
        target=target,
        strict_mode=strict_mode,
        intermediate=intermediate,
        enable_fusion=enable_fusion,
        plugin_settings=plugin_config or {},
      )

    if self.config.validation_report:
      self.semantics.load_validation_report(self.config.validation_report)

    self.source = self.config.effective_source
    self.target = self.config.effective_target
    self.strict_mode = self.config.strict_mode
    self.intermediate = intermediate or self.config.intermediate

    self.source_adapter = fw_registry.get_adapter(self.source)
    self.target_adapter = fw_registry.get_adapter(self.target)

    # Ensure built-in plugins are loaded now
    load_plugins()

  def parse(self, code: str) -> cst.Module:
    """Parses Python source code into a LibCST module."""
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Converts a LibCST module back to source code string."""
    return tree.code

  def _generate_snapshot(self, tree: cst.CSTNode, phase_label: str) -> None:
    """Generates a Mermaid graph snapshot and captures current source."""
    tracer = get_tracer()
    try:
      viz = MermaidGenerator()
      graph = viz.generate(tree)
      # Capture current state of the code for time-travel
      if isinstance(tree, cst.Module):
        source = self.to_source(tree)
      else:
        # Handle partial nodes safety
        source = "<Partial Node>"
      tracer.log_snapshot(phase_label, graph, source)
    except Exception as e:
      tracer.log_warning(f"Visualizer Error {phase_label}: {str(e)}")

  def _should_enforce_purity(self) -> bool:
    """Determines if Purity Analysis should run."""
    conf = self.semantics.get_framework_config(self.target)
    if conf:
      traits = conf.get("plugin_traits")
      if traits:
        if isinstance(traits, dict):
          return traits.get("enforce_purity_analysis", False)
        if isinstance(traits, PluginTraits):
          return traits.enforce_purity_analysis
        if hasattr(traits, "enforce_purity_analysis"):
          return getattr(traits, "enforce_purity_analysis", False)

    if self.target_adapter and hasattr(self.target_adapter, "plugin_traits"):
      return self.target_adapter.plugin_traits.enforce_purity_analysis

    return False

  def _run_mlir_roundtrip(self, tree: cst.Module, tracer: Any) -> cst.Module:
    """Executes the MLIR bridge pipeline."""
    return run_mlir_roundtrip(tree, tracer)

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.
    """
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Transpilation Pipeline", f"{self.source} -> {self.target}")

    # --- PHASE 1: INGESTION ---
    try:
      tree = ingest_code(code, self.source, self.target, self.source_adapter, tracer)
      self._generate_snapshot(tree, "After Ingestion")
    except Exception as e:
      tracer.end_phase()  # Root
      return ConversionResult(
        code=code,
        errors=[f"Parse Error: {e}"],
        success=False,
        trace_events=tracer.export(),
      )

    # Effective root for pruning (used for synthetic sources)
    effective_source_pruning_root = self.source
    if self.source == "latex_dsl":
      effective_source_pruning_root = "midl"

    # --- PHASE 2: EMISSION SHORT-CIRCUIT ---
    if self.target_adapter and hasattr(self.target_adapter, "create_emitter"):
      tracer.start_phase("Custom Emission", f"{self.target} Emitter")
      try:
        emitter = self.target_adapter.create_emitter()  # type: ignore
        current_source = self.to_source(tree)
        output_code = emitter.emit(current_source)
        tracer.log_mutation(
          "Transformed Emission",
          "(Python CST)",
          f"({self.target} Source)",
        )
        tracer.end_phase()
        tracer.end_phase()  # Root
        return ConversionResult(code=output_code, success=True, trace_events=tracer.export())
      except Exception as e:
        tracer.end_phase()
        tracer.end_phase()  # Root
        return ConversionResult(
          code="",
          errors=[f"Emission Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )

    if self.target == "mlir":
      try:
        emitter = PythonToMlirEmitter()
        mlir_cst = emitter.convert(tree)
        mlir_text = mlir_cst.to_text()
        return ConversionResult(code=mlir_text, success=True, trace_events=tracer.export())
      except Exception as e:
        return ConversionResult(code="", errors=[f"MLIR Error: {e}"], success=False)

    # --- PHASE 3: ANALYSIS ---
    errors_log: List[str] = []

    # Symbol Table Analysis (Updates symbol table for context)
    tracer.start_phase("Symbol Table Analysis", "Inferring types and scopes")
    symbol_table = None
    try:
      symbol_analyzer = SymbolTableAnalyzer(self.semantics)
      tree.visit(symbol_analyzer)
      symbol_table = symbol_analyzer.table
    except Exception as e:
      tracer.log_warning(f"Symbol Table analysis failed: {e}")
    tracer.end_phase()

    if self._should_enforce_purity():
      tracer.start_phase("Purity Analysis", "Scanning for side effects")
      purity_scanner = PurityScanner(semantics=self.semantics, source_fw=self.source)
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)

    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)

    self._generate_snapshot(tree, "After Analysis")

    # --- PHASE 4: OPTIMIZATION (Optional) ---
    if self.config.enable_graph_optimization:
      tracer.start_phase("Graph Optimization", "Fusion & Restructuring")
      try:
        from ml_switcheroo.core.graph import GraphExtractor
        from ml_switcheroo.core.graph_optimizer import GraphOptimizer
        from ml_switcheroo.core.graph_synthesizer import GraphSynthesizer

        # 1. Extract Logic
        extractor = GraphExtractor()
        tree.visit(extractor)
        graph = extractor.graph

        # 2. Optimize
        patterns = self.semantics.get_patterns()
        optimizer = GraphOptimizer(patterns)
        optimized_graph = optimizer.optimize(graph)

        # 3. Synthesize
        synthesizer = GraphSynthesizer(framework=self.target)
        src_code = synthesizer.generate(optimized_graph)

        # Update AST
        tree = self.parse(src_code)
        tracer.log_mutation("Graph Optimization", "AST", "Optimized AST")

      except Exception as e:
        tracer.log_warning(f"Optimization failed: {e}")

      tracer.end_phase()
      self._generate_snapshot(tree, "After Optimization")

    # --- PHASE 5: REWRITING ---
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")

    rewriter = PivotRewriter(self.semantics, self.config, symbol_table)
    tree = tree.visit(rewriter)

    tracer.end_phase()

    self._generate_snapshot(tree, "After Rewriting")

    # Only run MLIR bridge if specifically requested and valid
    if self.intermediate == "mlir":
      tree = self._run_mlir_roundtrip(tree, tracer)

    # --- PHASE 6: IMPORT FIXING (Optional) ---
    if self.config.enable_import_fixer and effective_source_pruning_root != self.target:
      tracer.start_phase("Import Fixer", "Resolving Dependencies")

      roots_to_prune = {effective_source_pruning_root}
      if self.source_adapter:
        if hasattr(self.source_adapter, "import_alias") and self.source_adapter.import_alias:
          roots_to_prune.add(self.source_adapter.import_alias[0].split(".")[0])

      usage_scanner = UsageScanner(effective_source_pruning_root)
      tree.visit(usage_scanner)
      should_preserve = usage_scanner.get_result()

      resolver = ImportResolver(self.semantics)
      plan = resolver.resolve(tree, self.target)

      fixer = ImportFixer(plan=plan, source_fws=roots_to_prune, preserve_source=should_preserve, target_fw=self.target)
      tree = tree.visit(fixer)
      tracer.end_phase()

      self._generate_snapshot(tree, "After Import Fixing")

    final_code = self.to_source(tree)

    # Linting
    if effective_source_pruning_root != self.target:
      linter = StructuralLinter(forbidden_roots={effective_source_pruning_root})
      lint_errors = linter.check(final_code)
      if lint_errors:
        errors_log.extend(lint_errors)
        tracer.start_phase("Structural Linter", "Detected Violations")
        for e in lint_errors:
          tracer.log_warning(e)
        tracer.end_phase()

    if "# <SWITCHEROO_FAILED_TO_TRANS>" in final_code:
      errors_log.append("Escape Hatches Detected")

    tracer.end_phase()
    return ConversionResult(
      code=final_code,
      errors=errors_log,
      success=True,
      trace_events=tracer.export(),
    )
