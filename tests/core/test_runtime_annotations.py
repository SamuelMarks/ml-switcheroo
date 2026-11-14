"""Unit and integration tests for `RuntimeAnnotationPass` and constructor concretization."""

from __future__ import annotations

from unittest import mock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.backends.python_strategies.init import (
  _extract_inferred_dim,
  generate_layer_init,
)
from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.runtime_annotations import (
  RuntimeAnnotationPass,
  SourceLazyConcretizer,
  _attach_comment_to_block_header,
  _attach_comment_to_line,
  _concretize_layer_call,
  _out_kwarg_for_in_kwarg,
  _replace_call_leaf,
  _resolve_target_traits,
)
from ml_switcheroo.core.runtime_models import (
  BranchResolutionRecord,
  ConstructorDimResolution,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)
from ml_switcheroo.core.runtime_tracer import run_on_device_trace
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


def _lightweight_semantics() -> SemanticsManager:
  """Create a lightweight `SemanticsManager` instance without disk I/O for unit tests.

  Returns:
      Lightweight `SemanticsManager`.
  """
  return SemanticsManager.empty()


def test_lazy_linear_to_flax_nnx_concretization() -> None:
  """Verify LazyLinear(10) -> nnx.Linear(in_features=16, out_features=10, rngs=rngs) with input (2, 16)."""
  src_code = """class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        h = self.fc(x)
        return h
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    input_shapes=[(2, 16)],
    inject_runtime_annotations=True,
    inject_shape_assertions=True,
  )
  trace = run_on_device_trace(src_code, "torch", "flax_nnx", config=cfg)
  assert trace.resolved_constructors["fc"].inferred_in_features == 16

  tree = cst.parse_module(src_code)
  tree = tree.visit(SourceLazyConcretizer(trace))
  assert "nn.Linear(16, 10)" in tree.code

  # Simulate target structure before RuntimeAnnotationPass
  target_tree = cst.parse_module(
    """class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.fc = nnx.Linear(16, 10, rngs=rngs)

    def __call__(self, x):
        h = self.fc(x)
        return h
"""
  )
  ctx = RewriterContext(_lightweight_semantics(), config=cfg, runtime_trace=trace)
  out_tree = RuntimeAnnotationPass().transform(target_tree, ctx)
  out_code = out_tree.code
  assert "nnx.Linear(in_features=16, out_features=10, rngs=rngs)" in out_code
  assert "# [on-device ctor] resolved in_features=16" in out_code
  assert "# [on-device fwd] shape=(2, 10) dtype=float32 | [bwd] grad=(2, 10)" in out_code
  assert "chex.assert_shape(h, (2, 10))" in out_code
  assert "import chex" in out_code


def test_flatten_to_lazy_linear_4d_input_resolution() -> None:
  """Verify Flatten -> LazyLinear(10) with 4D input (2, 3, 4, 4) resolves in_features=48."""
  src_code = """class ConvHead(nn.Module):
    def __init__(self):
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc(x)
        return x
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="torch",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 4, 4)],
    inject_runtime_annotations=True,
    inject_shape_assertions=True,
  )
  trace = run_on_device_trace(src_code, "torch", "torch", config=cfg)
  assert trace.resolved_constructors["fc"].inferred_in_features == 48

  ctx = RewriterContext(_lightweight_semantics(), config=cfg, runtime_trace=trace)
  out_tree = RuntimeAnnotationPass().transform(cst.parse_module(src_code), ctx)
  out_code = out_tree.code
  assert "in_features=48, out_features=10" in out_code
  assert "assert tuple(x.shape) == (2, 48)" in out_code
  assert "assert tuple(x.shape) == (2, 10)" in out_code


def test_dynamic_mask_branch_loop_and_praxis_create_child() -> None:
  """Verify dynamic boolean mask comments, branch/loop header comments, and Praxis create_child."""
  code = """class PraxisModel:
    def setup(self):
        self.create_child('fc', Linear(output_dims=6))

    def __call__(self, x):
        self.attr_val = x
        x += 1.0
        if x.sum() > -999.0:
            x = self.fc(x)
        while x.sum() > -999.0:
            x = x + 0.1
        mask = x > 0.0
        y = x[mask]
        return y
"""
  cfg = RuntimeConfig(
    source_framework="paxml",
    target_framework="paxml",
    enable_on_device_execution=True,
    input_shapes=[(2, 8)],
    inject_runtime_annotations=True,
    inject_shape_assertions=False,
  )
  trace = run_on_device_trace(code, "paxml", "paxml", config=cfg)
  ctx = RewriterContext(_lightweight_semantics(), config=cfg, runtime_trace=trace)
  out_code = RuntimeAnnotationPass().transform(cst.parse_module(code), ctx).code
  assert "input_dims=8" in out_code
  assert "# [on-device ctor] resolved in_features=8" in out_code
  assert "# [on-device branch] taken=True" in out_code
  assert "# [on-device loop] iterations=" in out_code
  assert "[dynamic-shape: resolved on-device (seed=42)]" in out_code


def test_annotation_helpers_and_edge_branches() -> None:
  """Verify all helper functions and edge branches in runtime_annotations.py."""
  sem = _lightweight_semantics()
  sem.framework_configs["jax"] = {"traits": {"lazy_layer_dim_arg_map": {"Linear": "input_dims"}}}
  ctx_custom = RewriterContext(sem, config=RuntimeConfig(target_framework="jax"))
  with mock.patch("ml_switcheroo.core.rewriter.passes.runtime_annotations.get_adapter", return_value=None):
    t_custom = _resolve_target_traits(ctx_custom)
    assert t_custom.lazy_layer_dim_arg_map == {"Linear": "input_dims"}

    sem.framework_configs["jax"] = {}
    t_empty = _resolve_target_traits(ctx_custom)
    assert t_empty.lazy_layer_dim_arg_map == {}

  # _attach_comment_to_line when line already has a comment
  line_with_comment = cst.parse_statement("""x = 1  # existing
""")
  assert isinstance(line_with_comment, cst.SimpleStatementLine)
  assert isinstance(line_with_comment, cst.SimpleStatementLine)
  updated_line = _attach_comment_to_line(line_with_comment, "# added")
  assert "existing  # added" in cst.Module([updated_line]).code

  # _attach_comment_to_block_header on SimpleStatementSuite (non-IndentedBlock)
  simple_suite = cst.SimpleStatementSuite(body=[cst.Pass()])
  assert _attach_comment_to_block_header(simple_suite, "# comment") is simple_suite

  # _replace_call_leaf on Name, Attribute, and other BaseExpression
  assert cst.Module([]).code_for_node(_replace_call_leaf(cst.Name("LazyLinear"), "Linear")) == "Linear"
  assert cst.Module([]).code_for_node(_replace_call_leaf(cst.parse_expression("nn.LazyLinear"), "Linear")) == "nn.Linear"
  sub_expr = cst.parse_expression("fn()[0]")
  assert _replace_call_leaf(sub_expr, "Linear") is sub_expr

  # _out_kwarg_for_in_kwarg
  assert _out_kwarg_for_in_kwarg("input_dims") == "output_dims"
  assert _out_kwarg_for_in_kwarg("in_channels") == "out_channels"
  assert _out_kwarg_for_in_kwarg("in_features") == "out_features"

  # _concretize_layer_call edge cases:
  res = ConstructorDimResolution(layer_name="fc", op_type="Dense", inferred_in_features=12, out_features=6)
  traits_flax = StructuralTraits(
    layer_namespace_prefix="nnx",
    lazy_layer_dim_arg_map={"Linear": "in_features", "Conv2d": "in_channels"},
  )
  # 1. Dense mapped to Linear with 1 positional arg
  c_dense = cst.parse_expression("Dense(6)")
  assert isinstance(c_dense, cst.Call)
  assert "in_features=12, out_features=6" in cst.Module([]).code_for_node(
    _concretize_layer_call(c_dense, res, traits_flax)
  )
  # 2. Existing in_features kwarg replaced
  c_existing = cst.parse_expression("Linear(in_features=4, out_features=6)")
  assert isinstance(c_existing, cst.Call)
  assert "in_features=12" in cst.Module([]).code_for_node(_concretize_layer_call(c_existing, res, traits_flax))
  # 3. >2 positional args (e.g. Conv2d(3, 8, 5))
  res_conv = ConstructorDimResolution(layer_name="c", op_type="Conv2d", inferred_in_features=3, out_features=8)
  c_conv = cst.parse_expression("Conv2d(1, 8, 5)")
  assert isinstance(c_conv, cst.Call)
  assert "Conv2d(3, 8, 5)" in cst.Module([]).code_for_node(_concretize_layer_call(c_conv, res_conv, traits_flax))
  # 4. Call with non-Name/Attribute func and unmapped op
  c_weird = cst.parse_expression("fn()(6)")
  assert isinstance(c_weird, cst.Call)
  assert _concretize_layer_call(c_weird, res, traits_flax).deep_equals(c_weird)
  # 5. Call with 0 positional args (only keyword args)
  c_kw_only = cst.parse_expression("Linear(out_features=6)")
  assert isinstance(c_kw_only, cst.Call)
  assert "in_features=12" in cst.Module([]).code_for_node(_concretize_layer_call(c_kw_only, res, traits_flax))

  # SourceLazyConcretizer with non-Call assign, non-self assign, bare Name LazyLinear, non-Lazy call, and unmapped self attr
  src_tree = cst.parse_module(
    """class M:
    def __init__(self):
        x = 1
        other.fc = LazyLinear(6)
        self.untracked = LazyLinear(6)
        self.fc = LazyLinear(6)
        self.norm = LayerNorm(6)
        self.weird = fn()(6)
"""
  )
  trace_obj = RuntimeTraceReport(
    executed=True,
    resolved_constructors={"fc": res, "norm": res, "weird": res},
  )
  concretized_src = src_tree.visit(SourceLazyConcretizer(trace_obj)).code
  assert "self.fc = Linear(12, 6)" in concretized_src

  # RuntimeAnnotationPass when runtime_trace is None or executed=False
  ctx_none = RewriterContext(sem, runtime_trace=None)
  assert RuntimeAnnotationPass().transform(src_tree, ctx_none) is src_tree

  # RuntimeAnnotationPass when inject_runtime_annotations=False, multi-stmt line, subscript target, unmatched var
  cfg_no_ann = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    inject_runtime_annotations=False,
    inject_shape_assertions=True,
  )
  tel_x = NodeRuntimeTelemetry(node_id="n0", var_name="x", op_type="Add", output_shape=[2, 4])
  trace_min = RuntimeTraceReport(
    executed=True,
    node_telemetry={"n0": tel_x},
    resolved_constructors={"fc": res},
    resolved_branches=[BranchResolutionRecord(lineno=1, condition_src="x", taken_branch="false")],
    loop_trip_counts={},
  )
  tree_edge = cst.parse_module(
    """import chex
class M:
    def __init__(self):
        self.fc = Linear(6)
        self.other = Linear(6)
        self.create_child('fc', Linear(output_dims=6))
        self.create_child('other', Linear(output_dims=6))
    def forward(self, x):
        a = 1; b = 2
        arr[0] = x
        x = x + 1
        x = x + 2
        unmatched = x
        unmatched += 1
        if x:
            pass
        if x:
            pass
        while False:
            pass
        return x
"""
  )
  ctx_min = RewriterContext(sem, config=cfg_no_ann, runtime_trace=trace_min)
  out_edge = RuntimeAnnotationPass().transform(tree_edge, ctx_min).code
  assert "chex.assert_shape(x, (2, 4))" in out_edge

  # Also test taken_branch == "false" with inject_runtime_annotations=True and second if/while exceeding cursor
  cfg_ann_only = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    inject_runtime_annotations=True,
    inject_shape_assertions=False,
  )
  ctx_ann = RewriterContext(sem, config=cfg_ann_only, runtime_trace=trace_min)
  out_ann = RuntimeAnnotationPass().transform(tree_edge, ctx_ann).code
  assert "# [on-device branch] taken=False" in out_ann


def test_python_strategy_init_and_engine_integration() -> None:
  """Verify generate_layer_init with runtime_trace/shape_metadata and ASTEngine.run()."""
  node_lazy = LogicalNode(id="fc1", op_type="LazyLinear", attributes={"arg_0": "10"})
  res_fc1 = ConstructorDimResolution(layer_name="fc1", op_type="LazyLinear", inferred_in_features=16, out_features=10)
  trace = RuntimeTraceReport(executed=True, resolved_constructors={"fc1": res_fc1})

  stmt = generate_layer_init(node_lazy, "flax_nnx", None, runtime_trace=trace)
  code = cst.Module([stmt]).code
  assert "in_features=16" in code

  # Test _extract_inferred_dim from node.shape_metadata
  node_meta = LogicalNode(id="fc2", op_type="Linear", attributes={"in_features": "?"}, shape_metadata=(2, 32))
  assert _extract_inferred_dim(node_meta, None) == 32
  stmt_meta = generate_layer_init(node_meta, "torch", None, runtime_trace=None)
  assert "in_features=32" in cst.Module([stmt_meta]).code

  # Test _extract_inferred_dim when shape_metadata has non-int or is absent, or existing_val is already concrete
  node_concrete = LogicalNode(id="fc1", op_type="Linear", attributes={"in_features": "16", "out_features": "10"})
  stmt_conc = generate_layer_init(node_concrete, "torch", None, runtime_trace=trace)
  assert "in_features=16" in cst.Module([stmt_conc]).code

  node_sym = LogicalNode(id="fc2", op_type="Linear", shape_metadata=("B", "D"))
  assert _extract_inferred_dim(node_sym, None) is None
  node_empty = LogicalNode(id="fc2", op_type="Linear", shape_metadata=())
  assert _extract_inferred_dim(node_empty, None) is None
  assert _extract_inferred_dim(LogicalNode(id="fc3", op_type="Linear"), None) is None

  # Test ASTEngine._run_rewriter_pipeline and _run_compiler_pipeline with enable_on_device_execution=True
  sem = _lightweight_semantics()
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    enable_import_fixer=False,
    input_shapes=[(2, 8)],
  )
  engine = ASTEngine(semantics=sem, config=cfg)
  res = engine.run(
    """class Net:
    def __init__(self):
        self.fc = nn.LazyLinear(4)
    def forward(self, x):
        return self.fc(x)
"""
  )
  assert res.runtime_trace is not None
  assert res.runtime_trace.executed is True
  assert "in_features=8" in res.code

  res_fn = engine.run(
    """def forward(x):
    return x + 1
"""
  )
  assert res_fn.runtime_trace is not None
  assert res_fn.runtime_trace.executed is True

  cfg_comp = RuntimeConfig(
    source_framework="torch",
    target_framework="html",
    enable_on_device_execution=True,
    input_shapes=[(2, 4)],
  )
  engine_comp = ASTEngine(semantics=sem, config=cfg_comp)
  res_comp = engine_comp.run(
    """def forward(x):
    return x + 1
"""
  )
  assert res_comp.runtime_trace is not None
  assert res_comp.runtime_trace.executed is True
