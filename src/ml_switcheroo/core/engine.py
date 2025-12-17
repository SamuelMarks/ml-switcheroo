"""
Orchestration Engine for AST Transformations.

This module uses the configured `effective_target` to drive the compilation process.
By resolving 'Flavours' (e.g. `flax_nnx`), it ensures specific structural traits
are loaded by the `PivotRewriter` and `Analysis` passes.
"""

import libcst as cst
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ml_switcheroo.core.tracer import reset_tracer, get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.import_fixer import ImportFixer

# Import Linter
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.analysis.dependencies import DependencyScanner
from ml_switcheroo.analysis.lifecycle import InitializationTracker
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.frameworks as fw_registry


class ConversionResult(BaseModel):
  """
  Structured result of a single file conversion.
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

    Populates `self.source` and `self.target` with the effective (flavour-resolved)
    framework keys to ensure specialized adapters are used.
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

    # Flavour Support: Use 'flow-specific' keys
    self.source = self.config.effective_source
    self.target = self.config.effective_target
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

    Args:
        code: The input source string.

    Returns:
        ConversionResult: Object containing transformed code and error logs.
    """
    reset_tracer()
    tracer = get_tracer()

    # Log the effective translation path (e.g. torch -> flax_nnx)
    root_phase = tracer.start_phase("Transpilation Pipeline", f"{self.source} -> {self.target}")

    try:
      tracer.start_phase("Preprocessing", "Parsing & Analysis")
      tree = self.parse(code)
      tracer.log_mutation("Module", "(Raw Source)", "(AST Parsed)")
    except Exception as e:
      return ConversionResult(code=code, errors=[f"Parse Error: {e}"], success=False, trace_events=tracer.export())

    errors_log = []

    # Pass 0a: Purity Analysis
    # Check against JAX-like targets (jax, flax_nnx, paxml)
    if "jax" in self.target or "flax" in self.target or "pax" in self.target:
      tracer.start_phase("Purity Check", "Scanning for side-effects")
      purity_scanner = PurityScanner(semantics=self.semantics, source_fw=self.source)
      tree = tree.visit(purity_scanner)
      tracer.end_phase()

    # Pass 0b: Lifecycle Analysis
    lifecycle_tracker = InitializationTracker()
    tree.visit(lifecycle_tracker)

    if lifecycle_tracker.warnings:
      for warning in lifecycle_tracker.warnings:
        errors_log.append(f"Lifecycle Warning: {warning}")

    # Pass 0c: Dependency Analysis
    # Use effective source to ignore framework imports correctly
    dep_scanner = DependencyScanner(self.semantics, self.source)
    tree.visit(dep_scanner)
    if dep_scanner.unknown_imports:
      sorted_deps = sorted(list(dep_scanner.unknown_imports))
      msg = f"Warning: Unmapped 3rd-party dependencies detected: {', '.join(sorted_deps)}"
      errors_log.append(msg)

    tracer.end_phase()

    # Pass 1: Semantic Pivot (Rewriter)
    tracer.start_phase("Rewrite Engine", "Visitor Traversal")
    rewriter = PivotRewriter(self.semantics, self.config)
    tree = tree.visit(rewriter)
    tracer.end_phase()

    # Pass 2: Import Scaffolding
    # If effective root differs, we run fixer.
    roots_to_prune = {self.source}

    if self.source != self.target:
      tracer.start_phase("Import Fixer", "Resolving Dependencies")

      # Resolve Effective Roots for Pruning (Inheritance Aware)
      adapter = fw_registry.get_adapter(self.source)

      if adapter:
        # Add alias root (e.g. flax.nnx -> flax)
        if hasattr(adapter, "import_alias") and adapter.import_alias:
          roots_to_prune.add(adapter.import_alias[0].split(".")[0])

        # Add parent frameworks recursively (e.g. flax_nnx -> jax)
        if hasattr(adapter, "inherits_from") and adapter.inherits_from:
          parent = adapter.inherits_from
          roots_to_prune.add(parent)

          # Recurse one level up for parent alias
          parent_adp = fw_registry.get_adapter(parent)
          if parent_adp and hasattr(parent_adp, "import_alias"):
            roots_to_prune.add(parent_adp.import_alias[0].split(".")[0])

      # Convert set to sorted list for determinism
      source_roots = sorted(list(roots_to_prune))

      # Sub-step 2a: Check usage of ANY source root aliases
      should_preserve = False
      for root in source_roots:
        scanner = UsageScanner(root)
        tree.visit(scanner)
        if scanner.get_result():
          should_preserve = True
          break

      # Sub-step 2b: Retrieve import maps for effective target
      submodule_map = self.semantics.get_import_map(self.target)
      alias_map = self.semantics.get_framework_aliases()

      # Sub-step 2c: Fix imports
      fixer = ImportFixer(
        source_fws=source_roots,
        target_fw=self.target,
        submodule_map=submodule_map,
        alias_map=alias_map,
        preserve_source=should_preserve,
      )
      tree = tree.visit(fixer)
      tracer.end_phase()

    final_code = self.to_source(tree)

    # Pass 3: Structural Linter (Post-Conversion Validation)
    # Ensure no forbidden roots remain in the output
    if self.source != self.target:
      tracer.start_phase("Structural Linter", "Verifying Cleanup")
      linter = StructuralLinter(forbidden_roots=roots_to_prune)
      lint_errors = linter.check(final_code)

      if lint_errors:
        for err in lint_errors:
          full_err = f"Lint Violation: {err}"
          errors_log.append(full_err)
          tracer.log_warning(full_err)

      tracer.end_phase()

    # Scan for failure markers to populate the error report
    if "# <SWITCHEROO_FAILED_TO_TRANS>" in final_code:
      count = final_code.count("# <SWITCHEROO_FAILED_TO_TRANS>")
      errors_log.append(f"{count} block(s) marked for manual review (Escape Hatch).")

    tracer.end_phase()
    return ConversionResult(code=final_code, errors=errors_log, success=True, trace_events=tracer.export())
