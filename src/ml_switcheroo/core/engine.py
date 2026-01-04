"""
Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, the primary driver for the transpilation process.
It coordinates the various analysis and transformation passes required to convert code
from a source framework to a target framework.

Pipeline:

1.  **Ingestion**: Adapter Hooks (LaTeX) or Standard Parsing (Python/MLIR/TikZ).
2.  **Emission**: Adapter Hooks (LaTeX) or Standard Emission (Python/MLIR/TikZ).
3.  **Analysis**: Symbol Table Inference, Purity, Lifecycle, Dependency checks.
4.  **Transformation**: Rewriting via PivotRewriter.
5.  **Refinement**: Import Fixing and Linting.
"""

from typing import Any, Dict, Optional

import libcst as cst

import ml_switcheroo.frameworks as fw_registry
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.analysis.lifecycle import InitializationTracker
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir_bridge import run_mlir_roundtrip
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.core.tikz.analyser import GraphExtractor
from ml_switcheroo.core.tikz.emitter import TikzEmitter
from ml_switcheroo.core.tikz.synthesizer import GraphSynthesizer
from ml_switcheroo.core.tracer import get_tracer, reset_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import PluginTraits
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.utils.visualizer import MermaidGenerator


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

    Args:
        semantics: The Knowledge Base manager. If None, a new default manager is loaded.
        config: A specific RuntimeConfig object. If provided, overrides single attributes.
        source: Source framework key (e.g. 'torch').
        target: Target framework key (e.g. 'jax').
        strict_mode: If True, fail on unmapped APIs instead of preserving them.
        enable_fusion: If True, performs graph-level fusion pass.
        plugin_config: Dictionary of settings passed to plugins.
        intermediate: Optional intermediate representation ('mlir', 'tikz') or None.
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
    # Prioritize explicit argument if provided, else check config
    self.intermediate = intermediate or self.config.intermediate

    self.source_adapter = fw_registry.get_adapter(self.source)
    self.target_adapter = fw_registry.get_adapter(self.target)

    # Ensure built-in plugins are loaded now, so that hooks are available during rewrite
    # This prevents "Missing required plugin" errors in programmatic usage.
    load_plugins()

  def parse(self, code: str) -> cst.Module:
    """
    Parses Python source code into a LibCST module.

    Args:
        code: Source code string.

    Returns:
        A parsed CST object.
    """
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """
    Converts a LibCST module back to source code string.

    Args:
        tree: The modified CST module.

    Returns:
        The generated Python code.
    """
    return tree.code

  def _generate_snapshot(self, tree: cst.CSTNode, phase_label: str) -> None:
    """
    Generates a Mermaid graph snapshot of the current AST state for the tracer.

    Args:
        tree: The AST to visualize.
        phase_label: Description of the current phase (e.g. "Before Pivot").
    """
    tracer = get_tracer()
    try:
      viz = MermaidGenerator()
      graph = viz.generate(tree)
      tracer.log_snapshot(phase_label, graph)
    except Exception as e:
      tracer.log_warning(f"Visualizer Error {phase_label}: {str(e)}")

  def _should_enforce_purity(self) -> bool:
    """
    Determines if Purity Analysis should run based on target traits.

    Returns:
        True if the target framework requires functional purity (e.g. JAX).
    """
    conf = self.semantics.get_framework_config(self.target)
    if conf:
      traits = conf.get("plugin_traits")
      if traits:
        if isinstance(traits, dict):
          return traits.get("enforce_purity_analysis", False)
        if isinstance(traits, PluginTraits):
          return traits.enforce_purity_analysis
        return getattr(traits, "enforce_purity_analysis", False)

    if self.target_adapter and hasattr(self.target_adapter, "plugin_traits"):
      return self.target_adapter.plugin_traits.enforce_purity_analysis

    return False

  def _run_mlir_roundtrip(self, tree: cst.Module, tracer: Any) -> cst.Module:
    """
    Wrapper to execute CST -> MLIR Text -> CST pipeline.

    This method is maintained for backward compatibility with tests which expect
    it to exist as a private method of the Engine class. It delegates to the
    module-level `run_mlir_roundtrip` function.

    Args:
        tree: The Python CST.
        tracer: The active trace logger.

    Returns:
        A reconstructed Python CST (via MLIR).
    """
    return run_mlir_roundtrip(tree, tracer)

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.

    Steps:

    1.  **Ingestion**: Parse source (Python/MLIR/LaTeX) to AST.
    2.  **Short-Circuit**: If target is non-Python (MLIR/TikZ/Latex), emit immediately.
    3.  **Analysis**:
        - Run Symbol Table Analyzer for type inference.
        - Run Purity, Lifecycle, and Dependency checks.
    4.  **Transformation**: Rewriting via PivotRewriter.
    5.  **Refinement**: Run `ImportFixer` and `StructuralLinter` to clean results.

    Args:
        code: The source code string to transpile.

    Returns:
        A `ConversionResult` containing the output code, errors, and execution trace.
    """
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Transpilation Pipeline", f"{self.source} -> {self.target}")

    try:
      tree = ingest_code(code, self.source, self.target, self.source_adapter, tracer)
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

    # 1. Adapter Hook (e.g. LatexEmitter)
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

    # 2. MLIR Emission
    if self.target == "mlir":
      tracer.start_phase("MLIR Emission", "CST -> MLIR Text")
      try:
        emitter = PythonToMlirEmitter()
        mlir_cst = emitter.convert(tree)
        mlir_text = mlir_cst.to_text()
        tracer.log_mutation("Emission", "(Python CST)", mlir_text)
        tracer.end_phase()
        tracer.end_phase()
        return ConversionResult(code=mlir_text, success=True, trace_events=tracer.export())
      except Exception as e:
        tracer.end_phase()
        tracer.end_phase()
        return ConversionResult(
          code="",
          errors=[f"MLIR Emit Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )

    # 3. TikZ Emission
    if self.target == "tikz":
      tracer.start_phase("TikZ Emission", "CST -> Logical Graph -> TikZ Text")
      try:
        extractor = GraphExtractor()
        tree.visit(extractor)
        emitter = TikzEmitter()
        tikz_code = emitter.emit(extractor.graph)
        tracer.log_mutation("Emission", "(Python CST)", "(TikZ Source)")
        tracer.end_phase()
        tracer.end_phase()
        return ConversionResult(code=tikz_code, success=True, trace_events=tracer.export())
      except Exception as e:
        tracer.end_phase()
        tracer.end_phase()
        return ConversionResult(
          code="",
          errors=[f"TikZ Emit Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )

    # --- PHASE 3: ANALYSIS & REWRITING ---
    errors_log = []

    # 1. Symbol Table Analysis (New Step)
    tracer.start_phase("Symbol Table Analysis", "Inferring types and scopes")
    try:
      symbol_analyzer = SymbolTableAnalyzer(self.semantics)
      tree.visit(symbol_analyzer)
      symbol_table = symbol_analyzer.table
    except Exception as e:
      tracer.log_warning(f"Symbol Table analysis failed: {e}")
      symbol_table = None
    tracer.end_phase()

    if self._should_enforce_purity():
      tracer.start_phase("Purity Check", "Scanning for side-effects")
      purity_scanner = PurityScanner(semantics=self.semantics, source_fw=self.source)
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)
    if lifecycle_tracker.warnings:
      errors_log.extend([f"Lifecycle: {w}" for w in lifecycle_tracker.warnings])

    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)
    if dep_scanner.unknown_imports:
      errors_log.append(f"Deps: {dep_scanner.unknown_imports}")

    self._generate_snapshot(tree, "AST Before Pivot")

    # --- Graph Fusion Optimization Pass ---
    if self.config.enable_fusion:
      tracer.start_phase("Graph Optimization", "Pattern-Based Fusion")
      try:
        # 1. Extract Logic Graph
        extractor = GraphExtractor()
        tree.visit(extractor)
        logical_graph = extractor.graph

        if logical_graph.nodes:
          # 2. Run Fusion
          optimizer = GraphOptimizer(patterns=self.semantics.get_patterns())
          optimized_graph = optimizer.optimize(logical_graph)

          # 3. Synthesize Code (Replaces Rewriter Logic)
          # Note: Synthesizer needs framework target 'jax' or 'torch'
          synth_target = "jax" if self.target in ["jax", "flax", "flax_nnx"] else "torch"
          synthesizer = GraphSynthesizer(framework=self.target)  # Pass actual target for better alias handling
          generated_code = synthesizer.generate(optimized_graph)

          tracer.log_mutation("Fusion Synthesis", "(Original AST)", generated_code)

          # Re-parse generated code back to CST for ImportFixer/Linting
          tree = self.parse(generated_code)

      except Exception as e:
        tracer.log_warning(f"Optimization pass failed: {e}")
        # Fallback to standard flow (continue with original tree)

      tracer.end_phase()

    # Standard AST Rewriter (Only run if fusion didn't entirely replace pipeline or if fallback needed)
    # If fusion ran, 'tree' is now the optimized tree.
    # We run rewriter anyway to handle nuanced expression rewrites not captured by graph extraction.
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    # Inject symbol_table for intelligent rewriting
    rewriter = PivotRewriter(self.semantics, self.config, symbol_table=symbol_table)
    tree = tree.visit(rewriter)
    tracer.end_phase()

    self._generate_snapshot(tree, "AST After Pivot")

    # Only run MLIR bridge if specifically requested via config
    if self.intermediate == "mlir":
      tree = self._run_mlir_roundtrip(tree, tracer)

    # Import Fixing logic
    if effective_source_pruning_root != self.target:
      roots_to_prune = {effective_source_pruning_root}
      if self.source_adapter:
        if hasattr(self.source_adapter, "import_alias") and self.source_adapter.import_alias:
          roots_to_prune.add(self.source_adapter.import_alias[0].split(".")[0])

      tracer.start_phase("Import Fixer", "Resolving Dependencies")
      submodule_map = self.semantics.get_import_map(self.target)
      alias_map = self.semantics.get_framework_aliases()

      usage_scanner = UsageScanner(effective_source_pruning_root)
      tree.visit(usage_scanner)
      should_preserve = usage_scanner.get_result()

      fixer = ImportFixer(
        sorted(list(roots_to_prune)),
        self.target,
        submodule_map,
        alias_map,
        should_preserve,
      )
      tree = tree.visit(fixer)
      tracer.end_phase()

    final_code = self.to_source(tree)

    if effective_source_pruning_root != self.target:
      tracer.start_phase("Structural Linter", "Verifying Cleanup")
      # Recalculate forbidden set
      linter_forbidden = {effective_source_pruning_root}
      if self.source_adapter and hasattr(self.source_adapter, "import_alias") and self.source_adapter.import_alias:
        linter_forbidden.add(self.source_adapter.import_alias[0].split(".")[0])

      linter = StructuralLinter(forbidden_roots=linter_forbidden)
      lint_errors = linter.check(final_code)
      if lint_errors:
        errors_log.extend(lint_errors)
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
