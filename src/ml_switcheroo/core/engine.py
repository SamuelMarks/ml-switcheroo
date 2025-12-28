"""
Orchestration Engine for AST Transformations.

This module provides the `ASTEngine`, the primary driver for the transpilation process.
It coordinates the various analysis and transformation passes required to convert code
from a source framework to a target framework.

The Engine pipeline consists of:

1.  **Preprocessing**: Parsing source code into a LibCST tree.
2.  **Safety Analysis**:

    *   **Purity**: Checking for side effects if targeting functional frameworks (JAX).
    *   **Lifecycle**: Verifying variable initialization.
    *   **Dependencies**: Scanning for unmapped 3rd-party imports.

3.  **Semantic Pivoting**: Executing the `PivotRewriter` to map API calls and structure.
    *Creates AST Snapshots for visualization before and after this phase using Mermaid.*

4.  **Post-processing**:

    *   **Import Fixer**: Injecting necessary imports and removing unused source imports.
    *   **Structural Linting**: Verifying no artifacts from the source framework remain.

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
    semantics: SemanticsManager = None,
    config: Optional[RuntimeConfig] = None,
    # Legacy parameters for backward compat
    source: Optional[str] = None,
    target: Optional[str] = None,
    strict_mode: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
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

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.

    Passes performed:

    1.  Parse.
    2.  Purity Scan (if targeting traits.enforce_purity_analysis).
    3.  Lifecycle Analysis (Init/Forward mismatch).
    4.  Dependency Scan (Checking 3rd party libs).
    5.  Pivot Rewrite (The main transformation).
        *Includes visual snapshots before and after.*
    6.  Import Fixer (Injecting new imports, pruning old ones).
        *Checks for lingering usage before pruning.*
    7.  Structural Linting (Verifying output cleanliness).

    Args:
        code (str): The input source string.

    Returns:
        ConversionResult: Object containing transformed code and error logs.
    """
    reset_tracer()
    tracer = get_tracer()

    print(f"[Engine] Starting run: {self.source} -> {self.target} (Strict: {self.strict_mode})")

    _root_phase = tracer.start_phase("Transpilation Pipeline", f"{self.source} -> {self.target}")

    try:
      tracer.start_phase("Preprocessing", "Parsing & Analysis")
      tree = self.parse(code)
      tracer.log_mutation("Module", "(Raw Source)", "(AST Parsed)")
    except Exception as e:
      return ConversionResult(
        code=code,
        errors=[f"Parse Error: {e}"],
        success=False,
        trace_events=tracer.export(),
      )

    errors_log = []

    # Pass 0a: Purity Analysis
    # Controlled dynamically by Framework Traits (e.g. JAX/Flax opt-in)
    if self._should_enforce_purity():
      tracer.start_phase("Purity Check", "Scanning for side-effects")
      purity_scanner = PurityScanner(semantics=self.semantics, source_fw=self.source)
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    # Pass 0b: Lifecycle Analysis
    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)
    if lifecycle_tracker.warnings:
      errors_log.extend([f"Lifecycle: {w}" for w in lifecycle_tracker.warnings])

    # Pass 0c: Dependency Analysis
    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)
    if dep_scanner.unknown_imports:
      errors_log.append(f"Deps: {dep_scanner.unknown_imports}")

    tracer.end_phase()

    # --- VISUALIZER HOOK 1 ---
    self._generate_snapshot(tree, "AST Before Pivot")

    # Pass 1: Semantic Pivot (Rewriter)
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    rewriter = PivotRewriter(self.semantics, self.config)
    tree = tree.visit(rewriter)
    tracer.end_phase()

    # --- VISUALIZER HOOK 2 ---
    self._generate_snapshot(tree, "AST After Pivot")

    # Pass 2: Import Scaffolding
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

      fixer = ImportFixer(sorted(list(roots_to_prune)), self.target, submodule_map, alias_map, should_preserve)
      tree = tree.visit(fixer)
      tracer.end_phase()

    final_code = self.to_source(tree)

    # Pass 3: Linter
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
    return ConversionResult(code=final_code, errors=errors_log, success=True, trace_events=tracer.export())
