"""Comprehensive unit tests exercising all domain exception guards and edge branches."""

from __future__ import annotations

import ast
import json
import types
import typing
from pathlib import Path
from unittest import mock

import libcst as cst
import numpy as np
import pytest
from ml_switcheroo_ir import DType
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.analysis.audit import CoverageScanner
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer
from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.config import LossReduction, RuntimeConfig
from ml_switcheroo.core.compiler.backends.compiler_bridge import compile_logical_graph_with_backend
from ml_switcheroo.core.compiler.backends.cpp.cst import BinaryExpression, CppModule, Identifier
from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser
from ml_switcheroo.core.compiler.backends.cpp.transformer import CppCSTTransformer
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.ir import IrParseError, IrPythonParser
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser, RdnaToken, RdnaTransformer
from ml_switcheroo.core.compiler.frontends.semantic_parser import _SemanticTransformer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, topological_sort
from ml_switcheroo.core.cst.base import CSTNode, CSTTransformer, CSTVisitor
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.core.hooks import HookContext, clear_hooks, register_hook
from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.ir_bridge import (
  ForwardGraphBuildRequest,
  _eval_proxy_ast_expr,
  build_hydrated_forward_graph,
  render_logical_graph_mermaid,
  synthesize_backward_vjp_graph,
  trace_with_zero_proxy_tensors,
)
from ml_switcheroo.core.markup.cst import MarkupNode, TriviaNode
from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.rewriter.calls.dispatch import _check_rule_condition, evaluate_dispatch_rules
from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy
from ml_switcheroo.core.rewriter.calls.transformers import MacroSubstitutionTransformer, rewrite_as_macro
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
from ml_switcheroo.core.rewriter.passes.runtime_annotations import RuntimeAnnotationTransformer
from ml_switcheroo.core.rewriter.patcher import GraphPatcher, PatchAction, ReplaceAction
from ml_switcheroo.core.runtime_tracer import _invoke_codegen, run_on_device_trace
from ml_switcheroo.core.tikz.parser import TikzParser, TikzTransformer
from ml_switcheroo.core.tracer import TraceLogger
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.exceptions import (
  BackendEmissionError,
  ConfigurationError,
  CSTValidationError,
  DialectLoweringError,
  GraphExtractionError,
  IngestionParseError,
  MacroExpansionError,
  PluginExecutionError,
  PreambleInjectionError,
  SemanticLoadError,
  SemanticsResolutionError,
  SnapshotGroundingError,
  SnapshotLoadError,
  SyntaxParseError,
  TopologicalSortError,
  TranslationError,
  VerificationExecutionError,
)
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.importers.array_api_reader import ArrayApiReader
from ml_switcheroo.plugins.state_flag_injection import capture_eval_state
from ml_switcheroo.plugins.static_unroll import LoopVarReplacer, unroll_static_loops
from ml_switcheroo.semantics.file_loader import parse_docstring_sections
from ml_switcheroo.semantics.index_builder import IndexBuilder
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_frameworks, merge_overlay_data, merge_tier_data
from ml_switcheroo.semantics.resolution import SemanticResolver
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.sphinx_ext.autogen_ops import generate_op_docs
from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner, _evaluate_shape_calc
from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.utils.formatting import MarkupVisitor
from ml_switcheroo.utils.visualizer import MermaidGenerator

_NL = chr(10)


def test_audit_and_symbol_table_domain_exceptions() -> None:
  """Verify ``SnapshotGroundingError`` in ``CoverageScanner`` and ``CSTValidationError`` in ``SymbolTableAnalyzer``."""
  sem = SemanticsManager.empty()
  sem.data["BadOp"] = {"variants": "not_a_dict"}
  sem._reverse_index["torch.bad_op"] = ("BadOp", sem.data["BadOp"])
  scanner = CoverageScanner(sem, allowed_roots={"torch"})
  with pytest.raises(SnapshotGroundingError, match="Expected dict"):
    cst.parse_module("torch.bad_op(x)").visit(scanner)

  analyzer = SymbolTableAnalyzer(sem)
  cls_src = f"class C:{_NL}  pass{_NL}"
  fn_src = f"def f():{_NL}  pass{_NL}"
  cls_node = cst.ensure_type(cst.parse_statement(cls_src), cst.ClassDef)
  fn_node = cst.ensure_type(cst.parse_statement(fn_src), cst.FunctionDef)
  with pytest.raises(CSTValidationError, match="ClassDef"):
    analyzer.leave_ClassDef(cls_node)
  with pytest.raises(CSTValidationError, match="FunctionDef"):
    analyzer.leave_FunctionDef(fn_node)


def test_cli_and_sphinx_configuration_errors() -> None:
  """Verify ``ConfigurationError`` in ``WeightScriptGenerator`` and ``generate_op_docs``."""
  sem = SemanticsManager.empty()
  gen = WeightScriptGenerator(sem, RuntimeConfig())
  gen.source_adapter = None
  with pytest.raises(ConfigurationError, match="both source_adapter and target_adapter"):
    gen._generate_script([])

  with pytest.raises(ConfigurationError, match="srcdir"):
    generate_op_docs("not_a_sphinx_app")


def test_compiler_parsers_and_topological_sort_domain_exceptions() -> None:
  """Verify ``TopologicalSortError``, ``SyntaxParseError``, ``DialectLoweringError``, and ``BackendEmissionError``."""
  cyclic = LogicalGraph(
    name="cyclic",
    nodes={
      "a": LogicalNode(id="a", op_type="Relu", inputs=["b"]),
      "b": LogicalNode(id="b", op_type="Relu", inputs=["a"]),
    },
  )
  with (
    mock.patch("ml_switcheroo.core.compiler.ir._ir_topological_sort", side_effect=ValueError("cycle")),
    pytest.raises(TopologicalSortError, match="cycle"),
  ):
    topological_sort(cyclic)

  with mock.patch("importlib.import_module", return_value=types.ModuleType("empty_reg")):
    out = compile_logical_graph_with_backend(LogicalGraph(name="g"), "edge_wasm")
    assert "(module" in out

  cpp_p = CppParser("int x = 1;")
  with (
    mock.patch.object(cpp_p.transformer, "transform", return_value=Identifier(name="x")),
    pytest.raises(SyntaxParseError, match="Expected CppModule"),
  ):
    cpp_p.parse()

  class BadCppTransformer(CppCSTTransformer):
    """Test C++ transformer returning ``CppModule`` for ``Identifier``."""

    def visit_Identifier(self, node: Identifier) -> CppModule:
      """Return mismatched ``CppModule`` to trigger ``DialectLoweringError``.

      Args:
          node: Input ``Identifier`` node.

      Returns:
          Mismatched ``CppModule``.
      """
      return CppModule()

  bin_expr = BinaryExpression(left=Identifier(name="x"), operator="+", right=Identifier(name="y"))
  with pytest.raises(DialectLoweringError, match="Expected Expression"):
    BadCppTransformer().visit(bin_expr)

  with pytest.raises(IrParseError, match="Unsupported IR AST expression"):
    IrPythonParser._eval_ast_expr(ast.UnaryOp(op=ast.Invert(), operand=ast.Constant(value=1)), {})
  with pytest.raises(IrParseError, match="Non-callable target"):
    IrPythonParser._eval_ast_expr(ast.parse("(1)()", mode="eval").body, {})
  tuple_ir_src = _NL.join(
    [
      "import ml_switcheroo_ir as sw_ir",
      "def build_graph():",
      "  a, b = [1, 2]",
      "  return sw_ir.LogicalGraph(name='g', nodes={}, inputs=[], outputs=[])",
    ]
  )
  parsed_tuple_ir = IrPythonParser().parse(tuple_ir_src)
  assert isinstance(parsed_tuple_ir, LogicalGraph)

  sass_p = NvidiaSassParser("FADD R0, R1, R2;")
  with (
    mock.patch.object(sass_p.transformer, "transform", return_value="not_a_module"),
    pytest.raises(SyntaxParseError, match="Expected NvidiaSassModule"),
  ):
    sass_p.parse()

  bad_rdna_tok = RdnaToken("IDENT", "x99")
  bad_rdna_tok.leading_trivia = []
  with pytest.raises(SyntaxParseError, match="Invalid RDNA register"):
    RdnaTransformer().register([bad_rdna_tok])

  rdna_p = RdnaParser("v_add_f32 v0, v1, v2")
  with (
    mock.patch.object(rdna_p.transformer, "transform", return_value="not_rdna_module"),
    pytest.raises(SyntaxParseError, match="Expected RdnaModule"),
  ):
    rdna_p.parse()

  with pytest.raises(SyntaxParseError, match="Expected SemanticMarker"):
    _SemanticTransformer().marker_start([None, None])

  mlir_p = MlirParser('%0 = "sw.constant"() {value = 1} : () -> i32')
  with (
    mock.patch.object(mlir_p.transformer, "transform", return_value="not_module_node"),
    pytest.raises(SyntaxParseError, match="Expected ModuleNode"),
  ):
    mlir_p.parse()

  tikz_src = chr(92) + "begin{tikzpicture}" + chr(92) + "node (a) {x};" + chr(92) + "end{tikzpicture}"
  with (
    mock.patch.object(TikzTransformer, "transform", return_value="not_tikz_graph"),
    pytest.raises(SyntaxParseError, match="Expected TikzGraph"),
  ):
    TikzParser(tikz_src).parse()


def test_isa_synthesizer_and_cst_base_validation_errors() -> None:
  """Verify ``BackendEmissionError`` in ``IsaSynthesizer`` and ``CSTValidationError`` in CST/Markup visitors."""
  with (
    mock.patch.object(NvidiaSassSynthesizer, "_is_isa_semantics", return_value=False),
    pytest.raises(BackendEmissionError, match="non-None SemanticsManager"),
  ):
    NvidiaSassSynthesizer(semantics=SemanticsManager.empty())

  synth = NvidiaSassSynthesizer(semantics=SemanticsManager.empty())
  with (
    mock.patch.object(synth, "_is_target_node", return_value=False),
    pytest.raises(BackendEmissionError, match="Expected IsaNode"),
  ):
    synth._append_node([], NvidiaSassComment(text="ok"))

  class NodeA(CSTNode):
    """First test CST node."""

    def to_text(self) -> str:
      """Render ``NodeA`` text.

      Returns:
          Literal ``'a'``.
      """
      return "a"

  class NodeB(CSTNode):
    """Second test CST node."""

    def to_text(self) -> str:
      """Render ``NodeB`` text.

      Returns:
          Literal ``'b'``.
      """
      return "b"

  vis = CSTVisitor()
  vis.register_visitor(NodeA, lambda _n: None)
  with pytest.raises(CSTValidationError, match="Expected NodeA"):
    vis._visitors[NodeA](NodeB())

  tr = CSTTransformer()
  tr.register_transformer(NodeA, lambda n: n)
  with pytest.raises(CSTValidationError, match="Expected NodeA"):
    tr._transformers[NodeA](NodeB())

  class OtherMarkupNode(MarkupNode):
    """Distinct ``MarkupNode`` subclass for testing ``MarkupVisitor`` validation."""

    def emit(self, indent_level: int = 0) -> str:
      """Emit placeholder markup string.

      Args:
          indent_level: Indentation level.

      Returns:
          Empty string.
      """
      return ""

  m_vis = MarkupVisitor()
  m_vis.register_visitor(TriviaNode, lambda _n: None)
  with pytest.raises(CSTValidationError, match="Expected TriviaNode"):
    m_vis._visitors[TriviaNode](OtherMarkupNode())


def test_engine_ghost_ingestion_and_ir_bridge_edges() -> None:
  """Verify ``TranslationError``, ``IngestionParseError``, ``GraphExtractionError``, and ``ir_bridge`` branches."""
  sem = SemanticsManager.empty()
  engine = ASTEngine(semantics=sem, config=RuntimeConfig(source_framework="torch", target_framework="jax"))

  with mock.patch("ml_switcheroo.core.strategies.STRATEGY_REGISTRY", []):
    res = engine.run("x = 1")
    assert not res.success
    assert any("No conversion strategy" in err for err in res.errors)

  with (
    mock.patch(
      "ml_switcheroo.core.compiler.frontends.python.PythonFrontend.parse_to_graph",
      return_value=None,
    ),
    pytest.raises(TranslationError, match="failed to produce a LogicalGraph"),
  ):
    engine._run_compiler_pipeline("x = 1", tracer=TraceLogger())

  ghost_ref = GhostInspector.inspect(123, "pkg.const")
  assert ghost_ref.name == "const"

  class BadCustomAdapter:
    """Adapter returning a custom parser whose ``parse()`` yields a non-``cst.Module``."""

    def create_parser(self, code: str) -> BadCustomAdapter:
      """Return ``self`` as parser.

      Args:
          code: Input source string.

      Returns:
          ``self``.
      """
      return self

    def parse(self) -> cst.Module:
      """Return a ``MagicMock`` non-``Module`` object at runtime to trigger ``IngestionParseError``.

      Returns:
          Mock non-Module object.
      """
      fake_mod: cst.Module = mock.MagicMock()
      return fake_mod

  with pytest.raises(IngestionParseError, match=r"expected cst\.Module"):
    ingest_code("x", "custom", "jax", BadCustomAdapter(), TraceLogger())

  with pytest.raises(GraphExtractionError, match="Unsupported proxy AST expression"):
    _eval_proxy_ast_expr(ast.Constant(value=1), {})

  class IncompleteTracingModule(types.ModuleType):
    """Stub module exposing ``global_tracing_state = None`` and ``ProxyTensor = None``."""

    global_tracing_state: None = None
    ProxyTensor: None = None

  with mock.patch("importlib.import_module", return_value=IncompleteTracingModule("incomplete_tracing")):
    assert trace_with_zero_proxy_tensors("def f(x): return x", [(2, 4)]) is None

  unshaped_graph = LogicalGraph(
    name="unshaped",
    nodes={
      "x": LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType("float32")),
      "y": LogicalNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=None),
    },
    inputs=["x"],
    outputs=["y"],
  )

  class StubAstToIrModule(types.ModuleType):
    """Stub ``ast_to_ir`` module returning ``unshaped_graph``."""

    @staticmethod
    def parse_ast_to_ir(_code: str) -> LogicalGraph:
      """Return ``unshaped_graph``.

      Args:
          _code: Source code string.

      Returns:
          ``unshaped_graph``.
      """
      return unshaped_graph

  def _stub_import(name: str) -> types.ModuleType:
    """Return ``StubAstToIrModule`` for ``ast_to_ir`` and raise ``ImportError`` otherwise.

    Args:
        name: Module path string.

    Returns:
        ``StubAstToIrModule``.

    Raises:
        ImportError: When ``name`` is not ``ast_to_ir``.
    """
    if "ast_to_ir" in name:
      return StubAstToIrModule("ast_to_ir")
    raise ImportError(name)

  code_str = "def f(x): return x"
  parsed_tree = cst.parse_module(code_str)
  with mock.patch("ml_switcheroo.core.ir_bridge.importlib.import_module", side_effect=_stub_import):
    hydrated, _ = build_hydrated_forward_graph(
      ForwardGraphBuildRequest(
        code=code_str,
        tree=parsed_tree,
        input_shapes=[(2, 4)],
        output_shape=(2, 4),
        dtype="float32",
        config=RuntimeConfig(),
      )
    )
    assert hydrated.nodes["y"].shape_metadata == (2, 4)
    mermaid = render_logical_graph_mermaid(hydrated)
    assert "(2, 4)" in mermaid
    mermaid_unshaped = render_logical_graph_mermaid(
      LogicalGraph(
        name="u",
        nodes={
          "u0": LogicalNode(id="u0", op_type="Input", shape_metadata=None),
          "u1": LogicalNode(id="u1", op_type="Relu", inputs=["u0"], shape_metadata=None),
        },
      )
    )
    assert "u0 --> u1" in mermaid_unshaped

  scalar_fwd = LogicalGraph(
    name="scalar_fwd",
    nodes={"s0": LogicalNode(id="s0", op_type="Input", shape_metadata=(1,), dtype=DType("float32"))},
    inputs=["s0"],
    outputs=["s0"],
  )
  bwd_scalar = synthesize_backward_vjp_graph(scalar_fwd, LossReduction.MEAN, {}, "float32")
  assert isinstance(bwd_scalar, LogicalGraph)

  class StubAutodiffModule(types.ModuleType):
    """Stub ``autodiff`` module returning canonical ``scalar_fwd`` from ``grad``."""

    @staticmethod
    def grad(graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
      """Return ``graph`` unchanged as a valid ``LogicalGraph``.

      Args:
          graph: Input ``LogicalGraph``.
          wrt: Parameter IDs.
          output_id: Loss node ID.

      Returns:
          ``graph``.
      """
      return graph

  with mock.patch(
    "ml_switcheroo.core.ir_bridge.importlib.import_module",
    side_effect=lambda name: (
      StubAutodiffModule("autodiff") if "autodiff" in name else (_ for _ in ()).throw(ImportError(name))
    ),
  ):
    bwd_non_none = synthesize_backward_vjp_graph(scalar_fwd, LossReduction.MEAN, {}, "float32")
    assert bwd_non_none.name == "scalar_fwd"


def test_rewriter_dispatch_strategy_patcher_and_tracer_guards() -> None:
  """Verify rewriter dispatch, strategy, macro, patcher, and runtime tracer exception guards."""
  sem = SemanticsManager.empty()
  ctx = RewriterContext(semantics=sem, config=RuntimeConfig())
  call_node = cst.ensure_type(cst.parse_expression("foo(1)"), cst.Call)

  api_t = ApiTransformer(ctx)
  assert evaluate_dispatch_rules(api_t, call_node, [{}], {}) is None
  assert _check_rule_condition(call_node, {}) is False

  clear_hooks(mark_loaded=True)

  @register_hook("no_ctx_plugin")
  def _plugin(n: cst.Call, _c: HookContext) -> cst.CSTNode:
    """Return ``n`` unchanged.

    Args:
        n: Call node.
        _c: HookContext.

    Returns:
        Unchanged ``n``.
    """
    return n

  class ContextlessRewriter:
    """Minimal ``StrategyRewriterProtocol`` without ``context`` to test ``PluginExecutionError``."""

    source_framework: str = "torch"
    target_framework: str = "jax"
    strict_mode: bool = False

    def _is_module_alias(self, _node: cst.CSTNode) -> bool:
      """Return ``False``.

      Args:
          _node: CST node.

      Returns:
          ``False``.
      """
      return False

    def _report_failure(self, _msg: str) -> None:
      """No-op failure reporter.

      Args:
          _msg: Failure message.
      """

    def _report_warning(self, _msg: str) -> None:
      """No-op warning reporter.

      Args:
          _msg: Warning message.
      """

    def _handle_variant_imports(self, _mapping: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
      """No-op import handler.

      Args:
          _mapping: Variant mapping.
      """

    def _create_name_node(self, name: str) -> cst.Name:
      """Construct ``cst.Name``.

      Args:
          name: Identifier string.

      Returns:
          ``cst.Name``.
      """
      return cst.Name(name)

  ctxless = ContextlessRewriter()
  with pytest.raises(PluginExecutionError, match="HookContext"):
    execute_strategy(ctxless, call_node, call_node, {"requires_plugin": "no_ctx_plugin"}, {}, "Op")
  assert _apply_layout_permutation(call_node, {}, {}, ctxless) is call_node

  with (
    mock.patch.object(MacroSubstitutionTransformer, "leave_Name", return_value=cst.Pass()),
    pytest.raises(MacroExpansionError, match="expected BaseExpression"),
  ):
    rewrite_as_macro("{x}", [cst.Arg(value=cst.Integer("1"))], ["x"])

  sem.data["BadAux"] = {"variants": "not_a_dict"}
  sem._reverse_index["torch.bad_aux"] = ("BadAux", sem.data["BadAux"])
  aux_t = AuxiliaryTransformer(
    RewriterContext(semantics=sem, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  )
  dec = cst.Decorator(decorator=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("bad_aux")))
  assert aux_t.leave_Decorator(dec, dec) is dec

  rt_t = RuntimeAnnotationTransformer(
    RewriterContext(semantics=sem, config=RuntimeConfig(), runtime_trace=None),
    StructuralTraits(),
  )
  stmt_line = cst.SimpleStatementLine(body=[cst.Pass()])
  assert rt_t.leave_SimpleStatementLine(stmt_line, stmt_line) is stmt_line

  bad_replace = ReplaceAction.model_construct(node_id="n", new_node=None, is_init=False, input_vars=[])

  class StubEmitter:
    """Stub ``SnippetEmitterProtocol`` implementation."""

    def emit_init(self, node: LogicalNode) -> cst.SimpleStatementLine:
      """Return ``pass`` statement.

      Args:
          node: LogicalNode.

      Returns:
          ``SimpleStatementLine``.
      """
      return stmt_line

    def emit_call(self, node: LogicalNode, input_vars: list[str], output_var: str) -> cst.SimpleStatementLine:
      """Return ``pass`` statement.

      Args:
          node: LogicalNode.
          input_vars: Input variable names.
          output_var: Output variable name.

      Returns:
          ``SimpleStatementLine``.
      """
      return stmt_line

    def emit_expression(self, node: LogicalNode, input_vars: list[str]) -> cst.BaseExpression:
      """Return ``cst.Name('x')``.

      Args:
          node: LogicalNode.
          input_vars: Input variable names.

      Returns:
          ``cst.Name``.
      """
      return cst.Name("x")

  patch_plan: list[PatchAction] = [bad_replace]
  patcher = GraphPatcher(plan=patch_plan, provenance={"n": stmt_line}, emitter=StubEmitter())
  with pytest.raises(DialectLoweringError, match="ReplaceAction requires LogicalNode"):
    patcher._handle_node(stmt_line, stmt_line)

  with pytest.raises(BackendEmissionError, match="not found or not callable"):
    _invoke_codegen(types.ModuleType("empty_codegen"), "MissingGen", LogicalGraph(name="g"))

  empty_comp_mod = types.ModuleType("incomplete_comp")
  trace_src = f"def f(x):{_NL}  return x + 1{_NL}"
  with mock.patch("importlib.import_module", return_value=empty_comp_mod):
    rep = run_on_device_trace(trace_src, "torch", "jax")
    assert rep.executed is True


def test_semantics_plugins_importers_testing_and_utils_exceptions(tmp_path: Path) -> None:
  """Verify ``SnapshotLoadError``, ``SemanticLoadError``, ``SemanticsResolutionError``, ``PreambleInjectionError``, and ``VerificationExecutionError``.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  full_gen = MlirToPythonGenerator()
  raw_op = OperationNode.model_construct(
    name="func.func",
    name_trivia=None,
    attributes=[],
    regions=[],
    operands=[],
    results=[],
    result_types=[],
    leading_trivia=[],
    trailing_trivia=[],
  )
  fn_str_triv = full_gen.stmt_generator.convert_func_def(raw_op)
  assert fn_str_triv.name.value == "unknown_func"

  sem = SemanticsManager.empty()
  gen_file = tmp_path / "test_out.py"
  TestCaseGenerator(semantics_mgr=sem).generate({"Op": {"variants": "not_a_mapping"}}, gen_file)

  bad_snap = tmp_path / "bad_snap.json"
  bad_snap.write_text(json.dumps({"categories": {"math": [{"name": "abs", "params": "not_a_list"}]}}), encoding="utf-8")
  with pytest.raises(SnapshotLoadError, match="Expected list for 'params'"):
    ArrayApiReader().parse_snapshot(bad_snap)

  h_ctx = HookContext(semantics=sem, config=RuntimeConfig())
  h_ctx.metadata["state_flag_injection"] = "corrupted_store"
  eval_call = cst.ensure_type(cst.parse_expression("model.eval()"), cst.Call)
  with pytest.raises(PluginExecutionError, match="Corrupted state_flag_injection"):
    capture_eval_state(eval_call, h_ctx)

  for_src = f"for i in range(1):{_NL}  x = i{_NL}"
  for_node = cst.ensure_type(cst.parse_statement(for_src), cst.For)
  with mock.patch.object(cst.SimpleStatementLine, "visit", return_value=cst.Name("not_a_stmt")):
    assert LoopVarReplacer is not None
    with pytest.raises(PreambleInjectionError, match="expected BaseStatement"):
      unroll_static_loops(for_node, h_ctx)

  with (
    mock.patch("ml_switcheroo.semantics.file_loader.parse_docstring_with_griffe", return_value="not_a_list"),
    pytest.raises(SemanticLoadError, match="Expected list"),
  ):
    parse_docstring_sections("doc", "google")

  import ml_switcheroo.core.dsl as dsl_mod
  import ml_switcheroo.semantics.manager as mgr_mod

  sem.data["BadIdx"] = {"variants": "not_a_dict"}
  IndexBuilder.build_index(sem)
  with (
    mock.patch.object(dsl_mod.OperationDef, "model_dump", return_value={"variants": "not_a_dict"}),
    mock.patch.object(mgr_mod, "resolve_semantics_dir", return_value=tmp_path),
  ):
    fresh_sem = mgr_mod.SemanticsManager.empty()
    fresh_sem.update_definition(
      "CustomNonDict",
      {"operation": "CustomNonDict", "description": "d", "std_args": [], "variants": {}},
    )

  with pytest.raises(SemanticLoadError, match="Expected dict for framework"):
    merge_frameworks({}, {"bad_fw": "not_a_dict"})

  tier_data: dict[str, dict[str, JsonValue]] = {}
  merge_tier_data(
    data=tier_data,
    key_origins={},
    framework_configs={},
    new_content={"__frameworks__": "bad", "__patterns__": "bad"},
    tier=SemanticTier.EXTRAS,
    patterns=[],
  )

  with pytest.raises(SemanticLoadError, match="mappings must be a dict"):
    merge_overlay_data({}, {}, {}, {}, {"__framework__": "jax", "mappings": "bad"}, "jax.json")

  overlay_out: dict[str, dict[str, JsonValue]] = {}
  merge_overlay_data(
    overlay_out,
    {},
    {},
    {},
    {"__framework__": "jax", "mappings": {"MyOp": "non_dict_impl"}},
    "jax.json",
  )
  assert "MyOp" in overlay_out

  resolver = SemanticResolver(operation_maps={"BadOp": {"variants": "not_a_dict"}})
  with pytest.raises(SemanticsResolutionError, match="Expected dict for variants"):
    resolver.resolve("torch", "torch.add")
  with pytest.raises(SemanticsResolutionError, match="Expected dict for variants"):
    resolver.dispatch("BadOp", "jax")

  bisector = SemanticsBisector(runner=EquivalenceRunner())
  bisector.runner = None
  with pytest.raises(VerificationExecutionError, match="active EquivalenceRunner"):
    bisector.propose_fix("Add", {})

  arr = np.zeros((2, 4), dtype=np.float32)

  with pytest.raises(VerificationExecutionError, match="Unsupported attribute"):
    _evaluate_shape_calc("lambda x: x.ndim", {"x": arr})
  with pytest.raises(VerificationExecutionError, match="shape sequence"):
    _evaluate_shape_calc("lambda x: x[0]", {"x": 5})
  with pytest.raises(VerificationExecutionError, match="Slice bounds"):
    _evaluate_shape_calc("lambda x: [2, 4]['a':]", {"x": arr})
  with pytest.raises(VerificationExecutionError, match="Subscript index"):
    _evaluate_shape_calc("lambda x: [2, 4]['a']", {"x": arr})
  with pytest.raises(VerificationExecutionError, match="Unary '-'"):
    _evaluate_shape_calc("lambda x: -'a'", {"x": arr})

  with pytest.raises(TypeError, match="Expected callable"):
    CodeExtractor.extract_function_body("not_callable")

  m_gen = MermaidGenerator()
  m_gen.stack.clear()
  assign_node = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("x"))], value=cst.Integer("1"))
  m_gen.leave_Assign(assign_node)
