"""
Orchestration Engine for AST Transformations.

This module provides the `ASTEngine`, the primary driver for the transpilation process.
It coordinates the various analysis and transformation passes required to convert code
from a source framework to a target framework.

The Engine pipeline consists of:

1.  **Ingestion Phase**:
    - If `source="mlir"`, parses MLIR text and hydrates a Python AST.
    - Otherwise, parses Python source code into a LibCST tree.

2.  **MLIR Emission (Target="mlir")**:
    - If the target is "mlir", the pipeline short-circuits here.
    - The AST is converted to MLIR IR text and returned immediately.

3.  **Safety Analysis (Standard)**:
    - **Purity**: Checking for side effects if targeting functional frameworks (JAX).
    - **Lifecycle**: Verifying variable initialization.
    - **Dependencies**: Scanning for unmapped 3rd-party imports.

4.  **Semantic Pivoting**: Executing the `PivotRewriter` to map API calls and structure.
    *Creates AST Snapshots for visualization before and after this phase using Mermaid.*

5.  **Intermediate Representation (MLIR) Roundtrip** (Optional):
    - If `intermediate="mlir"`, the pivot result is converted to MLIR text,
      parsed back, and regenerated into Python to verify fidelity.

6.  **Post-processing**:
    - **Import Fixer**: Injecting necessary imports and removing unused source imports.
    - **Structural Linting**: Verifying no artifacts from the source framework remain.

The engine relies on `RuntimeConfig` to resolve 'Flavours' (e.g., distinguishing
`flax_nnx` from generic `jax`) to load the correct structural traits.
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

# Import Visualizer for snapshots
from ml_switcheroo.utils.visualizer import MermaidGenerator

# MLIR Bridge
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


class ConversionResult(BaseModel):
  """
  Structured result of a single file conversion.
  """

  code: str = Field(default="", description="The transformed source code.")
  errors: List[str] = Field(default_factory=list, description="A list of error messages or warnings.")
  success: bool = Field(
    default=True,
    description="True if the pipeline completed without critical failures.",
  )
  trace_events: List[Dict[str, Any]] = Field(default_factory=list, description="A log of internal trace events.")

  @property
  def has_errors(self) -> bool:
    """
    Returns True if any errors or warnings were recorded during conversion.

    Returns:
        bool: True if errors list is non-empty.
    """
    return len(self.errors) > 0


class ASTEngine:
  """
  The main compilation unit.

  This class encapsulates the state and logic required to transpile a single unit of code.
  It manages the lifecycle of the LibCST tree and the invocation of visitor passes.
  """

  def __init__(
    self,
    semantics: Optional[SemanticsManager] = None,
    config: Optional[RuntimeConfig] = None,
    # Legacy parameters for backward compat
    source: Optional[str] = None,
    target: Optional[str] = None,
    strict_mode: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
    intermediate: Optional[str] = None,
  ):
    """
    Initializes the Engine.

    Populates `self.source` and `self.target` with the effective (flavour-resolved)
    framework keys to ensure specialized adapters are used. If `config` is not provided,
    it builds one using the optional args or dynamic defaults if they are None.

    Args:
        semantics (SemanticsManager, optional): The knowledge base manager. Creates new if None.
        config (RuntimeConfig, optional): The runtime configuration object.
        source (str, optional): Legacy override for source framework key.
        target (str, optional): Legacy override for target framework key.
        strict_mode (bool): Legacy override for strict mode.
        plugin_config (Dict, optional): Legacy override for plugin settings.
        intermediate (str, optional): "mlir" to enable IR verification roundtrip.
    """
    self.semantics = semantics or SemanticsManager()

    if config:
      self.config = config
    else:
      self.config = RuntimeConfig.load(
        source=source,
        target=target,
        strict_mode=strict_mode,
        plugin_settings=plugin_config or {},
      )

    # Trigger Verification Gating if report provided
    if self.config.validation_report:
      self.semantics.load_validation_report(self.config.validation_report)

    # Flavour Support: Use 'flow-specific' keys
    self.source = self.config.effective_source
    self.target = self.config.effective_target
    self.strict_mode = self.config.strict_mode
    self.intermediate = intermediate

  def parse(self, code: str) -> cst.Module:
    """
    Parses source string into a LibCST Module.

    Args:
        code (str): Python source code.

    Returns:
        cst.Module: The parsed Abstract Syntax Tree.

    Raises:
        libcst.ParserSyntaxError: If the input code is invalid Python.
    """
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """
    Converts CST back to source string.

    Args:
        tree (cst.Module): The modified syntax tree.

    Returns:
        str: Generated Python code.
    """
    return tree.code

  def _generate_snapshot(self, tree: cst.CSTNode, phase_label: str) -> None:
    """
    Helper to generate and log an AST visualization snapshot.
    Includes verbose print logging visible in the UI console.
    """
    tracer = get_tracer()
    try:
      viz = MermaidGenerator()
      graph = viz.generate(tree)

      tracer.log_snapshot(phase_label, graph)
    except Exception as e:
      tracer.log_warning(f"Visualizer Error {phase_label}: {str(e)}")

      error_graph = (
        f'graph TD\nclassDef err fill:#ea4335,color:white,font-weight:bold;\nE["Visualizer Crash:<br/>{str(e)}"]:::err\n'
      )
      tracer.log_snapshot(f"{phase_label} (FAILED)", error_graph)

  def _should_enforce_purity(self) -> bool:
    """
    Determines if the target framework requires purity checks based on PluginTraits.
    Checks the SemanticsManager configuration first, and falls back to inspecting
    the live Adapter directly if configuration hasn't been hydrated.

    Returns:
        bool: True if the target framework requires functional purity checks.
    """
    # 1. Check Semantics Config (Hydrated from JSON)
    conf = self.semantics.get_framework_config(self.target)
    if conf:
      traits_data = conf.get("plugin_traits")
      if traits_data:
        if isinstance(traits_data, dict):
          return traits_data.get("enforce_purity_analysis", False)
        if isinstance(traits_data, PluginTraits):
          return traits_data.enforce_purity_analysis
        # Fallback for generic object access
        return getattr(traits_data, "enforce_purity_analysis", False)

    # 2. Fallback: Check Adapter Logic directly (Source of Truth)
    # This covers cases where manager loading might lag behind code changes or in pure unit tests
    adapter = fw_registry.get_adapter(self.target)
    if adapter and hasattr(adapter, "plugin_traits"):
      return adapter.plugin_traits.enforce_purity_analysis

    return False

  def _run_mlir_roundtrip(self, tree: cst.Module, tracer: Any) -> cst.Module:
    """
    Executes CST -> MLIR Text -> CST pipeline for verification.
    """
    try:
      tracer.start_phase("MLIR Bridge", "CST -> MLIR Text -> CST")

      # 1. Emit
      emitter = PythonToMlirEmitter()
      mlir_cst = emitter.convert(tree)
      mlir_text = mlir_cst.to_text()

      tracer.log_mutation("MLIR Generation", "(Python CST)", mlir_text)

      # 2. Parse Back
      parser = MlirParser(mlir_text)
      mlir_cst_restored = parser.parse()
      mlir_text_roundtrip = mlir_cst_restored.to_text()

      if mlir_text != mlir_text_roundtrip:
        tracer.log_warning("MLIR Text Roundtrip Mismatch")

      # 3. Generate Python
      generator = MlirToPythonGenerator()
      restored_tree = generator.generate(mlir_cst_restored)

      tracer.end_phase()
      return restored_tree

    except Exception as e:
      tracer.log_warning(f"MLIR Bridge Failed: {e}")
      tracer.end_phase()
      # Return original if bridge fails to ensure pipeline robustness
      return tree

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.

    Args:
        code (str): The input source string.

    Returns:
        ConversionResult: Object containing transformed code and error logs.
    """
    reset_tracer()
    tracer = get_tracer()

    print(f"[Engine] Starting run: {self.source} -> {self.target} (Strict: {self.strict_mode})")

    _root_phase = tracer.start_phase("Transpilation Pipeline", f"{self.source} -> {self.target}")

    tree: cst.Module

    # --- PHASE 1: INGESTION (Parsing/Hydration) ---
    if self.source == "mlir":
      tracer.start_phase("MLIR Ingest", "MLIR Text -> Python CST")
      try:
        parser = MlirParser(code)
        mlir_mod = parser.parse()
        gen = MlirToPythonGenerator()
        tree = gen.generate(mlir_mod)
        tracer.log_mutation("Ingestion", "(MLIR Text)", "(Python CST)")
      except Exception as e:
        return ConversionResult(
          code=code,
          errors=[f"MLIR Parse Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )
      tracer.end_phase()
    else:
      tracer.start_phase("Preprocessing", "Parsing & Analysis")
      try:
        tree = self.parse(code)
        tracer.log_mutation("Module", "(Raw Source)", "(AST Parsed)")
      except Exception as e:
        return ConversionResult(
          code=code,
          errors=[f"Parse Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )
      tracer.end_phase()

    # --- PHASE 2: SHORT CURCUIT (Target = MLIR) ---
    if self.target == "mlir":
      tracer.start_phase("MLIR Emission", "CST -> MLIR Text")
      try:
        emitter = PythonToMlirEmitter()
        mlir_cst = emitter.convert(tree)
        mlir_text = mlir_cst.to_text()
        tracer.log_mutation("Emission", "(Python CST)", mlir_text)
        tracer.end_phase()
        tracer.end_phase()  # End root phase
        return ConversionResult(code=mlir_text, success=True, trace_events=tracer.export())
      except Exception as e:
        return ConversionResult(
          code="",
          errors=[f"MLIR Emit Error: {e}"],
          success=False,
          trace_events=tracer.export(),
        )

    # --- PHASE 3: STANDARD REWRITING ---

    errors_log = []

    # Pass 3a: Purity Analysis (if targeting safe framework)
    if self._should_enforce_purity():
      tracer.start_phase("Purity Check", "Scanning for side-effects")
      purity_scanner = PurityScanner(semantics=self.semantics, source_fw=self.source)
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    # Pass 3b: Lifecycle Analysis
    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)
    if lifecycle_tracker.warnings:
      errors_log.extend([f"Lifecycle: {w}" for w in lifecycle_tracker.warnings])

    # Pass 3c: Dependency Analysis
    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)
    if dep_scanner.unknown_imports:
      errors_log.append(f"Deps: {dep_scanner.unknown_imports}")

    # --- VISUALIZER HOOK 1 ---
    self._generate_snapshot(tree, "AST Before Pivot")

    # Pass 4: Semantic Pivot (Rewriter)
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    rewriter = PivotRewriter(self.semantics, self.config)
    tree = tree.visit(rewriter)
    tracer.end_phase()

    # --- VISUALIZER HOOK 2 ---
    self._generate_snapshot(tree, "AST After Pivot")

    # Pass 5: MLIR Bridge (Optional Verification Roundtrip)
    if self.intermediate == "mlir":
      tree = self._run_mlir_roundtrip(tree, tracer)

    # Pass 6: Import Scaffolding
    roots_to_prune = {self.source}
    if self.source != self.target:
      tracer.start_phase("Import Fixer", "Resolving Dependencies")
      adapter = fw_registry.get_adapter(self.source)
      if adapter:
        if hasattr(adapter, "import_alias") and adapter.import_alias:
          roots_to_prune.add(adapter.import_alias[0].split(".")[0])
        if hasattr(adapter, "inherits_from") and adapter.inherits_from:
          roots_to_prune.add(adapter.inherits_from)

      submodule_map = self.semantics.get_import_map(self.target)
      alias_map = self.semantics.get_framework_aliases()

      # Check for lingering usage of source framework logic before instructing fixer to prune
      usage_scanner = UsageScanner(self.source)
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

    # Pass 7: Linter
    if self.source != self.target:
      tracer.start_phase("Structural Linter", "Verifying Cleanup")
      linter = StructuralLinter(forbidden_roots=roots_to_prune)
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
