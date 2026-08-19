"""Orchestration Engine for AST Transformations.

This module provides the ``ASTEngine``, generating code via:

 **Compiler Pipeline**: For ISA/Visuals (Source -> Graph -> Backend -> Target).

**Rewriter Pipeline**: For High-Level Frameworks (Source -> CST -> Pipeline(Structure, API, Aux) -> Target).

Supports optional **Graph-Guided Rewriting** (Loopback).
"""

from typing import Any, Dict, Optional, cast
import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult as ConversionResult
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
from ml_switcheroo.core.compiler.registry import (
  is_isa_target,
  is_isa_source,
  get_backend_class,
)
from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.frontends.sass import SassParser, SassLifter
from ml_switcheroo.core.compiler.frontends.rdna import RdnaParser, RdnaLifter
from ml_switcheroo.core.compiler.backend import CompilerBackend
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
    """Initializes the engine with semantics and configuration.

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
    """Executes the complete conversion pipeline on the input code.

    This method initiates the translation process from the source framework to
    the target framework. Depending on the configuration and targets, it
    routes the code through either the compiler pipeline (for ISAs or when
    sharding is enabled), a specialized stablehlo bypass, or the standard
    rewriter pipeline.

    Args:
        code: The input source code string to be converted.

    Returns:
        A ConversionResult object containing the converted code, success status,
        and any collected execution trace events or error messages.
    """
    reset_tracer()
    tracer = get_tracer()
    tracer.start_phase("Pipeline Start", f"{self.source} -> {self.target}")

    try:
      if is_isa_source(self.source) or is_isa_target(self.target) or self.config.enable_sharding:
        result = self._run_compiler_pipeline(code, tracer)
      elif self.target == "stablehlo":
        # special bypass for CST to MLIR string translation
        source_adapter = get_adapter(self.source)
        tree = ingest_code(code, self.source, self.target, source_adapter, tracer)
        from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter

        emitter = StableHloEmitter(self.semantics)
        mlir_code = emitter.convert(tree).to_text()
        result = ConversionResult(code=mlir_code, success=True, trace_events=tracer.export())
      else:
        result = self._run_rewriter_pipeline(code, tracer)

      tracer.end_phase()
      return result
    except Exception as e:
      import traceback

      traceback.print_exc()
      tracer.end_phase()
      return ConversionResult(
        code=code,
        errors=[f"Critical Failure: {str(e)}"],
        success=False,
        trace_events=tracer.export(),
      )

  def parse(self, code: str) -> cst.Module:
    """Parses a Python source code string into a LibCST Module.

    Args:
        code: The Python source code as a string.

    Returns:
        A libcst.Module representation of the input source code.
    """
    return cst.parse_module(code)

  def to_source(self, tree: cst.Module) -> str:
    """Converts a LibCST Module back into its source code string representation.

    Args:
        tree: The LibCST Module to be serialized.

    Returns:
        The generated source code as a string.
    """
    return tree.code

  def _graph_to_mermaid(self, tree: cst.CSTNode) -> str:
    """Generates a Mermaid diagram definition representing the CST structure.

    Args:
        tree: The LibCST node to visualize.

    Returns:
        A string containing the Mermaid diagram definition.
    """
    return MermaidGenerator().generate(tree)

  def _run_compiler_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    """Runs the Graph-based compiler pipeline for Instruction Set Architectures (ISAs).

    This pipeline handles conversion for targets like SASS, RDNA, or graph-level
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
      except Exception:
        # Fallback: Maybe it's valid python code already
        frontend = PythonFrontend(code)
        graph = frontend.parse_to_graph()

    else:
      # ISA Source (SASS/RDNA) logic
      if self.source == "sass":
        parser = SassParser(code)
        nodes = parser.parse().statements
        lifter: Any = SassLifter()
        graph = lifter.lift(nodes)
      elif self.source == "rdna":
        parser = RdnaParser(code)  # type: ignore
        nodes = parser.parse().statements
        lifter = RdnaLifter()
        graph = lifter.lift(nodes)
      elif self.source == "stablehlo":
        # StableHLO parses via ingest_code to Python CST, then frontend parses to graph
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
      from ml_switcheroo.core.compiler.sharding import ShardingInferencePass
      from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
      from ml_switcheroo.core.compiler.fusion import QKVFusionPass, QKVDefusionPass
      from ml_switcheroo.core.compiler.qwen_fusion import (
        SwiGLUFusionPass,
        SwiGLUDefusionPass,
        VisionPatchEmbeddingFusionPass,
        VisionPatchEmbeddingDefusionPass,
      )

      # Extract pre-existing constraints
      graph = ShardingExtractionPass().apply(graph)

      # Apply Defusions first to normalize
      graph = QKVDefusionPass().apply(graph)
      graph = SwiGLUDefusionPass().apply(graph)
      graph = VisionPatchEmbeddingDefusionPass().apply(graph)

      # Re-infer constraints and re-fuse for target
      if self.target in ["jax", "flax", "flax_nnx", "paxml"]:
        graph = ShardingInferencePass().apply(graph)
        graph = QKVFusionPass().apply(graph)
        graph = SwiGLUFusionPass().apply(graph)
        graph = VisionPatchEmbeddingFusionPass().apply(graph)
      else:
        graph = ShardingInferencePass().apply(graph)

    backend_cls = get_backend_class(self.target)
    if not backend_cls:
      raise ValueError(f"No backend found for {self.target}")

    if backend_cls.__name__ == "PythonBackend":
      backend = backend_cls(framework=self.target, semantics=self.semantics)  # type: ignore
    else:
      backend = cast(CompilerBackend, backend_cls(self.semantics))  # type: ignore

    output_code = backend.compile(graph)
    tracer.log_mutation("Codegen", "(Graph)", output_code)
    tracer.end_phase()
    return ConversionResult(code=output_code, success=True, trace_events=tracer.export())

  def _run_rewriter_pipeline(self, code: str, tracer: Any) -> ConversionResult:
    """Runs the structural rewriter pipeline with optional graph loopback optimization.

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
        from ml_switcheroo.core.graph_optimizer import GraphOptimizer
        from ml_switcheroo.core.compiler.differ import GraphDiffer
        from ml_switcheroo.core.rewriter.patcher import GraphPatcher
        from ml_switcheroo.core.compiler.backends.python_snippet import (
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

          if self.config.enable_sharding:
            from ml_switcheroo.core.compiler.sharding import ShardingInferencePass
            from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
            from ml_switcheroo.core.compiler.fusion import QKVFusionPass, QKVDefusionPass
            from ml_switcheroo.core.compiler.qwen_fusion import (
              SwiGLUFusionPass,
              SwiGLUDefusionPass,
              VisionPatchEmbeddingFusionPass,
              VisionPatchEmbeddingDefusionPass,
            )

            # Reverse translation: lift inline sharding out of AST into metadata
            optimized_graph = ShardingExtractionPass().apply(optimized_graph)

            # Normalize structure
            optimized_graph = QKVDefusionPass().apply(optimized_graph)
            optimized_graph = SwiGLUDefusionPass().apply(optimized_graph)
            optimized_graph = VisionPatchEmbeddingDefusionPass().apply(optimized_graph)

            # Forward translation: synthesize optimized blocks and metadata
            optimized_graph = ShardingInferencePass().apply(optimized_graph)
            if self.target in ["jax", "flax", "flax_nnx", "paxml"]:
              optimized_graph = QKVFusionPass().apply(optimized_graph)
              optimized_graph = SwiGLUFusionPass().apply(optimized_graph)
              optimized_graph = VisionPatchEmbeddingFusionPass().apply(optimized_graph)
              # Note: Torch keeps defused standard blocks, but gets sharding annotations

          # C. Differ
          differ = GraphDiffer()
          plan = differ.diff(original_graph, optimized_graph)

          # D. Patching
          if plan:
            emitter = PythonSnippetEmitter(framework=self.target)
            patcher = GraphPatcher(plan, provenance, emitter)  # type: ignore
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
      except Exception as e:
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
      plan = resolver.resolve(tree, self.target)  # type: ignore
      fixer = ImportFixer(
        plan=plan,  # type: ignore
        source_fws={
          self.config.source_framework,
          self.config.effective_source,
          self.semantics.get_framework_config(self.config.source_framework)
          .get("alias", {})
          .get("module", "")
          .split(".")[0],
          self.semantics.get_framework_config(self.config.effective_source)
          .get("alias", {})
          .get("module", "")
          .split(".")[0],
        }
        - {""},
        preserve_source=should_preserve,
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
    """Helper property to retrieve the strict mode setting from config.

    Returns:
        True if strict mode is enabled, False otherwise.
    """
    return self.config.strict_mode
