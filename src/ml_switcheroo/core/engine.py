"""
Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, generating code via:
1.  **Compiler Pipeline**: For ISA/Visuals (Source -> Graph -> Backend -> Target).
2.  **Rewriter Pipeline**: For High-Level Frameworks (Source -> CST -> Rewriter -> Target).
"""

from typing import Any, Dict, Optional
import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.core.scanners import UsageScanner
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.linter import StructuralLinter
from ml_switcheroo.core.tracer import get_tracer, reset_tracer
from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Compiler Components
from ml_switcheroo.compiler.registry import (
  is_isa_target,
  is_isa_source,
  get_backend_class,
)
from ml_switcheroo.compiler.frontends.python import PythonFrontend
from ml_switcheroo.compiler.frontends.sass import SassParser, SassLifter
from ml_switcheroo.compiler.frontends.rdna import RdnaParser, RdnaLifter
from ml_switcheroo.core.mlir_bridge import run_mlir_roundtrip


class ASTEngine:
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
    self.semantics = semantics or SemanticsManager()
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
        enable_fusion=enable_fusion,
        plugin_settings=plugin_config or {},
      )
    if self.config.validation_report:
      self.semantics.load_validation_report(self.config.validation_report)

    self.source = self.config.effective_source
    self.target = self.config.effective_target
    load_plugins()

  def run(self, code: str) -> ConversionResult:
    """
    Main execution point for the pipeline.
    Dispatches to either the Compiler (IR) or Rewriter (AST) path.
    """
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Pipeline Start", f"{self.source} -> {self.target}")

    try:
      if is_isa_source(self.source) or is_isa_target(self.target):
        result = self._run_compiler_pipeline(code, tracer)
      else:
        # Default for high-level frameworks
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
    """Parses python source into a CST (Helper for tests)."""
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Converts CST back to source code."""
    return tree.code

  def _run_mlir_roundtrip(self, input_obj: Any, tracer: Any) -> Any:
    if not isinstance(input_obj, cst.Module):
      tracer.log_warning("Skipping MLIR roundtrip: Input is not a CST Module.")
      return input_obj
    return run_mlir_roundtrip(input_obj, tracer)

  def _run_compiler_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    tracer.start_phase("Compiler Pipeline", f"{self.source}->Graph->{self.target}")
    graph = None
    if is_isa_source(self.source):
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
    else:
      frontend = PythonFrontend(code)
      graph = frontend.parse_to_graph()

    if self.config.enable_graph_optimization:
      from ml_switcheroo.core.graph_optimizer import GraphOptimizer

      tracer.start_phase("Optimization", "Fusion")
      patterns = self.semantics.get_patterns()
      optimizer = GraphOptimizer(patterns)
      graph = optimizer.optimize(graph)
      tracer.log_mutation("Graph Optimization", "(Graph)", "(Optimized Graph)")
      tracer.log_snapshot("After Optimization", "graph TD...", "...")
      tracer.end_phase()

    backend_cls = get_backend_class(self.target)
    if not backend_cls:
      raise ValueError(f"No backend found for {self.target}")

    if backend_cls.__name__ == "PythonBackend":
      backend = backend_cls(framework=self.target)
    else:
      backend = backend_cls(self.semantics)

    output_code = backend.compile(graph)
    tracer.log_mutation("Codegen", "(Graph)", output_code)
    tracer.end_phase()
    return ConversionResult(code=output_code, success=True, trace_events=tracer.export())

  def _run_rewriter_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    tracer.start_phase("Rewriter Pipeline", "AST Transformation")

    # 1. Ingestion
    from ml_switcheroo.frameworks.base import get_adapter

    source_adapter = get_adapter(self.source)
    tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
    tracer.log_snapshot("After Ingestion", self.to_source(tree), self.to_source(tree))
    tracer.log_snapshot("After Analysis", self.to_source(tree), self.to_source(tree))

    # 2. Rewriting
    rewriter = PivotRewriter(self.semantics, self.config)
    tree = tree.visit(rewriter)
    tracer.log_snapshot("After Rewriting", self.to_source(tree), self.to_source(tree))

    # 3. Import Fixing
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
        target_fw=self.target,
      )
      tree = tree.visit(fixer)
      tracer.log_snapshot("After Import Fixing", self.to_source(tree), self.to_source(tree))

    # 4. Optional MLIR
    if self.config.intermediate == "mlir":
      tree = self._run_mlir_roundtrip(tree, tracer)

    # 5. Emission
    final_code = tree.code
    target_adapter = get_adapter(self.target)
    if target_adapter and hasattr(target_adapter, "create_emitter"):
      try:
        emitter = target_adapter.create_emitter()
        if hasattr(emitter, "convert") and isinstance(tree, cst.Module):
          final_code = emitter.convert(tree).to_text()
        elif hasattr(emitter, "emit"):
          final_code = emitter.emit(final_code)
      except Exception as e:
        tracer.log_warning(f"Emitter output failed: {e}")

    # Trace for debugging logic if needed
    if self.target == "mlir" or self.target == "stablehlo":
      tracer.log_mutation("Final Emission", "(Python CST)", final_code)

    # 6. Checks
    errors = []

    # Check for Escape Hatches markers in output
    if EscapeHatch.START_MARKER in final_code:
      msg = "Escape Hatches Detected: Partial conversion. Inspect output for '# <SWITCHEROO...' blocks."
      errors.append(msg)
      tracer.log_warning(msg)

    if self.strict_mode and self.target not in ["mlir", "stablehlo", "latex_dsl", "tikz"]:
      # Start Phase to register in trace
      tracer.start_phase("Structural Linter", "Safety Verification")
      linter = StructuralLinter(forbidden_roots={self.source})
      list_errors = linter.check(final_code)
      if list_errors:
        tracer.log_warning(f"Linter errors: {list_errors}")
        errors.extend(list_errors)
      tracer.end_phase()

    tracer.end_phase()
    return ConversionResult(code=final_code, success=True, errors=errors, trace_events=tracer.export())

  @property
  def strict_mode(self) -> bool:
    return self.config.strict_mode
