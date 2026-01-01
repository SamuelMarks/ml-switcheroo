"""
Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, the primary driver for the transpilation process.
It coordinates the various analysis and transformation passes required to convert code
from a source framework to a target framework.

Pipeline:

1.  **Ingestion**: Adapter Hooks (LaTeX) or Standard Parsing (Python/MLIR/TikZ).
2.  **Emission**: Adapter Hooks (LaTeX) or Standard Emission (Python/MLIR/TikZ).
3.  **Transformation**: Purity, Lifecycle, Dependency checks, Rewriting, Import Fixing.
"""

import traceback
import libcst as cst
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ml_switcheroo.core.tracer import reset_tracer, get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.analysis.lifecycle import InitializationTracker
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import PluginTraits
import ml_switcheroo.frameworks as fw_registry

# Visualizer
from ml_switcheroo.utils.visualizer import MermaidGenerator

# MLIR Bridge
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator

# TikZ Bridge
from ml_switcheroo.core.tikz.parser import TikzParser
from ml_switcheroo.core.tikz.synthesizer import GraphSynthesizer
from ml_switcheroo.core.tikz.analyser import GraphExtractor
from ml_switcheroo.core.tikz.emitter import TikzEmitter


class ConversionResult(BaseModel):
  """
  Container for the results of a transpilation job.
  """

  code: str = Field(default="", description="The generated source code.")
  errors: List[str] = Field(default_factory=list, description="List of error messages encountered.")
  success: bool = Field(default=True, description="True if the pipeline completed without fatal crashes.")
  trace_events: List[Dict[str, Any]] = Field(default_factory=list, description="Execution trace log data.")

  @property
  def has_errors(self) -> bool:
    """Check if the result contains any error messages."""
    return len(self.errors) > 0


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
    Executes CST -> MLIR Text -> CST pipeline for verification.

    Used when intermediate="mlir" is selected to verify the structural integrity
    of the MLIR bridge.

    Args:
        tree: The Python CST.
        tracer: The active trace logger.

    Returns:
        A reconstructed Python CST (via MLIR).
    """
    try:
      tracer.start_phase("MLIR Bridge", "CST -> MLIR Text -> CST")
      emitter = PythonToMlirEmitter()
      mlir_cst = emitter.convert(tree)
      mlir_text = mlir_cst.to_text()
      tracer.log_mutation("MLIR Generation", "(Python CST)", mlir_text)

      parser = MlirParser(mlir_text)
      mlir_cst_restored = parser.parse()

      generator = MlirToPythonGenerator()
      restored_tree = generator.generate(mlir_cst_restored)
      tracer.end_phase()
      return restored_tree
    except Exception as e:
      tracer.log_warning(f"MLIR Bridge Failed: {e}")
      tracer.end_phase()
      return tree

  def _run_ingestion(self, code: str, tracer) -> Optional[cst.Module]:
    """
    Parses input code handles non-python sources via adapters.

    Supports:
    1. Adapter-specific parsers (e.g. LaTeX).
    2. MLIR text parsing.
    3. TikZ text parsing.
    4. Standard Python parsing.

    Args:
        code: Raw source code string.
        tracer: The trace logger instance.

    Returns:
        A validated Python LibCST Module.
    """
    # 1. Adapter Hook (e.g. LatexParser)
    if self.source_adapter and hasattr(self.source_adapter, "create_parser"):
      tracer.start_phase("Custom Ingest", f"{self.source} Parser")
      try:
        parser = self.source_adapter.create_parser(code)
        tree = parser.parse()
        tracer.log_mutation("Transformed Ingestion", "(Raw Source)", "(AST Parsed)")
        tracer.end_phase()
        return tree
      except Exception as e:
        tracer.end_phase()
        raise e

    # 2. MLIR source
    if self.source == "mlir":
      tracer.start_phase("MLIR Ingest", "MLIR Text -> Python CST")
      try:
        parser = MlirParser(code)
        mlir_mod = parser.parse()
        gen = MlirToPythonGenerator()
        tree = gen.generate(mlir_mod)
        tracer.log_mutation("Ingestion", "(MLIR Text)", "(Python CST)")
        tracer.end_phase()
        return tree
      except Exception as e:
        tracer.end_phase()
        raise e

    # 3. TikZ source
    if self.source == "tikz":
      tracer.start_phase("TikZ Ingest", "TikZ Text -> Logical Graph -> Python CST")
      try:
        parser = TikzParser(code)
        graph = parser.parse()
        synth_target = "jax" if self.target in ["jax", "flax", "flax_nnx"] else "torch"
        synthesizer = GraphSynthesizer(framework=synth_target)
        py_code = synthesizer.generate(graph, class_name="SwitcherooNet")
        tree = self.parse(py_code)
        tracer.log_mutation("Ingestion", "(TikZ Source)", f"(Python CST)\n{py_code}")
        tracer.end_phase()
        return tree
      except Exception as e:
        tracer.end_phase()
        raise e

    # 4. Standard Python
    tracer.start_phase("Preprocessing", "Parsing & Analysis")
    try:
      tree = self.parse(code)
      tracer.log_mutation("Transformed Module", "(Raw Source)", "(AST Parsed)")
      tracer.end_phase()
      return tree
    except Exception as e:
      tracer.end_phase()
      raise e

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.

    Steps:

    1.  **Ingestion**: Parse source (Python/MLIR/LaTeX) to AST.
    2.  **Short-Circuit**: If target is non-Python (MLIR/TikZ/Latex), emit immediately.
    3.  **Analysis**: Run Purity, Lifecycle, and Dependency checks.
    4.  **Rewriting**: Execute `PivotRewriter` to transform the AST.
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
      tree = self._run_ingestion(code, tracer)
    except Exception as e:
      tracer.end_phase()  # Root
      return ConversionResult(code=code, errors=[f"Parse Error: {e}"], success=False, trace_events=tracer.export())

    # Effective root for pruning (used for synthetic sources)
    effective_source_pruning_root = self.source
    if self.source == "latex_dsl":
      effective_source_pruning_root = "midl"

    # --- PHASE 2: EMISSION SHORT-CIRCUIT ---

    # 1. Adapter Hook (e.g. LatexEmitter)
    if self.target_adapter and hasattr(self.target_adapter, "create_emitter"):
      tracer.start_phase("Custom Emission", f"{self.target} Emitter")
      try:
        emitter = self.target_adapter.create_emitter()
        current_source = self.to_source(tree)
        output_code = emitter.emit(current_source)
        tracer.log_mutation("Transformed Emission", "(Python CST)", f"({self.target} Source)")
        tracer.end_phase()
        tracer.end_phase()  # Root
        return ConversionResult(code=output_code, success=True, trace_events=tracer.export())
      except Exception as e:
        tracer.end_phase()
        tracer.end_phase()  # Root
        return ConversionResult(code="", errors=[f"Emission Error: {e}"], success=False, trace_events=tracer.export())

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
        return ConversionResult(code="", errors=[f"MLIR Emit Error: {e}"], success=False, trace_events=tracer.export())

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
        return ConversionResult(code="", errors=[f"TikZ Emit Error: {e}"], success=False, trace_events=tracer.export())

    # --- PHASE 3: STANDARD REWRITING ---
    errors_log = []

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

    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    rewriter = PivotRewriter(self.semantics, self.config)
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

      fixer = ImportFixer(sorted(list(roots_to_prune)), self.target, submodule_map, alias_map, should_preserve)
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
    return ConversionResult(code=final_code, errors=errors_log, success=True, trace_events=tracer.export())
