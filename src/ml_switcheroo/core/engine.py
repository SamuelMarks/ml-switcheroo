"""
Orchestration Engine for AST Transformations.

This module provides the high-level `ASTEngine` class which parses source code,
applies the sequence of Transformer passes (Purity Analysis, Dependency Scan,
Rewriting, Import Fixing), and emits the final source code.

It returns structured `ConversionResult` objects to enable rich reporting.
"""

import libcst as cst
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ml_switcheroo.core.tracer import reset_tracer, get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.analysis.lifecycle import InitializationTracker
from ml_switcheroo.config import RuntimeConfig


class ConversionResult(BaseModel):
  """
  Structured result of a single file conversion.

  Attributes:
      code: The transpiled source code string.
      errors: List of error messages or warnings encountered (e.g. from Escape Hatch).
      success: True if the AST was parseable and processed without critical crashes.
  """

  code: str = ""
  errors: List[str] = Field(default_factory=list)
  success: bool = True
  trace_events: List[Dict[str, Any]] = Field(default_factory=list)

  @property
  def has_errors(self) -> bool:
    """Returns True if any errors were recorded."""
    return len(self.errors) > 0


class ASTEngine:
  """
  The main compilation unit.

  Attributes:
      semantics (SemanticsManager): Database of mappings.
      config (RuntimeConfig): The full configuration container (Frameworks, Strict Mode, Plugin Settings).
  """

  def __init__(
    self,
    semantics: SemanticsManager = None,
    config: Optional[RuntimeConfig] = None,
    # Legacy parameters for backward compat
    source: str = "torch",
    target: str = "jax",
    strict_mode: bool = False,
    plugin_config: Optional[Dict[str, Any]] = None,
  ):
    """
    Initializes the Engine.

    If a validation report is specified in the config, it is loaded into the
    semantics manager to gate rewrites.

    Args:
        semantics: The loaded SemanticsManager instance.
        config: A populated RuntimeConfig Pydantic model.
        source: Source framework name (legacy fallback).
        target: Target framework name (legacy fallback).
        strict_mode: Warning vs Silence flag (legacy fallback).
        plugin_config: Dict of plugin settings (legacy fallback).
    """
    self.semantics = semantics or SemanticsManager()

    if config:
      self.config = config
    else:
      self.config = RuntimeConfig(
        source_framework=source,
        target_framework=target,
        strict_mode=strict_mode,
        plugin_settings=plugin_config or {},
      )

    # Trigger Verification Gating if report provided
    if self.config.validation_report:
      self.semantics.load_validation_report(self.config.validation_report)

    self.source = self.config.source_framework
    self.target = self.config.target_framework
    self.strict_mode = self.config.strict_mode

  def parse(self, code: str) -> cst.Module:
    """Parses source string into a LibCST Module."""
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Converts CST back to source string."""
    return tree.code

  def run(self, code: str) -> ConversionResult:
    """
    Executes the full transpilation pipeline.

    Steps:
    0a. Purity Analysis (if targeting JAX): Flag unsafe side-effects.
    0b. Lifecycle Analysis: Detect dynamic attribute definition.
    0c. Dependency Analysis: Flag unmapped 3rd party libraries.
    1. Rewrite Call usages (Semantic Pivot).
    2. Scan for lingering source references.
    3. Feature-driven Import Fixing.

    Args:
        code: The input source string.

    Returns:
        ConversionResult: Object containing transformed code and error logs.
    """
    reset_tracer()
    tracer = get_tracer()
    root_phase = tracer.start_phase("Transpilation Pipeline", "Full Process")
    try:
      tracer.start_phase("Preprocessing", "Parsing & Analysis")
      tree = self.parse(code)
      tracer.log_mutation("Module", "(Raw Source)", "(AST Parsed)")
    except Exception as e:
      return ConversionResult(code=code, errors=[f"Parse Error: {e}"], success=False, trace_events=tracer.export())

    errors_log = []

    # Pass 0a: Purity Analysis (Specific to Functional Frameworks like JAX)
    if self.target == "jax":
      tracer.start_phase("Purity Check", "Scanning for side-effects")
      purity_scanner = PurityScanner()
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    # Pass 0b: Lifecycle Analysis (Detect dynamic state definition)
    # This is critical for converting Pythonic PyTorch to Static Graph JAX/Flax
    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)

    if lifecycle_tracker.warnings:
      for warning in lifecycle_tracker.warnings:
        errors_log.append(f"Lifecycle Warning: {warning}")

    # Pass 0c: Dependency Analysis (Validation)
    # Identifies imports like 'pandas', 'cv2' that aren't mapped in semantics.
    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)

    if dep_scanner.unknown_imports:
      # We treat these as warnings, not compilation failures, as the code might still run e.g. in eager mode
      sorted_deps = sorted(list(dep_scanner.unknown_imports))
      msg = f"Warning: Unmapped 3rd-party dependencies detected: {', '.join(sorted_deps)}"
      errors_log.append(msg)

    tracer.end_phase()

    # Pass 1: Semantic Pivot (Change usage)
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    rewriter = PivotRewriter(self.semantics, self.config)
    tree = tree.visit(rewriter)
    tracer.end_phase()

    # Pass 2: Import Scaffolding (Change imports)
    if self.config.source_framework != self.config.target_framework:
      tracer.start_phase("Import Fixer", "Resolving Dependencies")
      # Sub-step 2a: Check for lingering usages
      # Checks if source framework symbols remain after rewriting (e.g. Escape Hatches)
      scanner = UsageScanner(self.config.source_framework)
      tree.visit(scanner)
      should_preserve = scanner.get_result()

      # Sub-step 2b: Retrieve Data-Driven Import Maps & Alias Config
      submodule_map = self.semantics.get_import_map(self.config.target_framework)
      alias_map = self.semantics.get_framework_aliases()

      # Sub-step 2c: Fix imports
      fixer = ImportFixer(
        self.config.source_framework,
        self.config.target_framework,
        submodule_map=submodule_map,
        alias_map=alias_map,
        preserve_source=should_preserve,
      )
      tree = tree.visit(fixer)
      tracer.end_phase()

    final_code = self.to_source(tree)

    # Scan for failure markers to populate the error report
    if "# <SWITCHEROO_FAILED_TO_TRANS>" in final_code:
      count = final_code.count("# <SWITCHEROO_FAILED_TO_TRANS>")
      errors_log.append(f"{count} block(s) marked for manual review (Escape Hatch).")

    tracer.end_phase()
    return ConversionResult(code=final_code, errors=errors_log, success=True, trace_events=tracer.export())
