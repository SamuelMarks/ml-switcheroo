"""Unit tests for Phase 19: Input-Dependent `for` Loop Bounds (`range(x.shape[i])`) & Static Unrolling Interweaving.

Verifies:
- `for i in range(x.shape[1]):` and `for i in range(x.size(0)):` concretization in `SourceLazyConcretizer` and `RuntimeAnnotationTransformer`.
- Static unrolling via `unroll_static_loops` across `PyTorch -> JAX / Flax NNX / StableHLO` with `# [on-device unrolled iter=0/N]` comments and 100% trivia preservation.
- Imperative target (`PyTorch -> PyTorch`) preserving `for i in range(3):` with `# [on-device loop] iterations=3` header comment.
"""

from __future__ import annotations

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.runtime_annotations import (
  RuntimeAnnotationPass,
  SourceLazyConcretizer,
  _concretize_range_iter,
)
from ml_switcheroo.core.runtime_models import RuntimeTraceReport
from ml_switcheroo.core.runtime_tracer import _ForwardSymbolicExecutor
from ml_switcheroo.semantics.manager import SemanticsManager


def test_dynamic_for_loop_unrolling_pytorch_to_jax_flax_stablehlo() -> None:
  """Verify `for i in range(x.shape[1]):` concretizes and unrolls to JAX, Flax NNX, and StableHLO with trivia preserved."""
  code = """import torch

def forward(x):
    # Loop header leading trivia
    for i in range(x.shape[1]):  # inline loop header comment
        # Inside loop body comment
        x = torch.relu(x)  # inline body op comment
        x = torch.sin(x)
    return x
"""
  sem = SemanticsManager()

  for target_fw in ("jax", "flax_nnx"):
    cfg = RuntimeConfig(
      source_framework="torch",
      target_framework=target_fw,
      enable_on_device_execution=True,
      inject_runtime_annotations=True,
      input_shapes=[(2, 3)],
    )
    res = ASTEngine(semantics=sem, config=cfg).run(code)
    assert res.success
    assert res.runtime_trace is not None
    assert res.runtime_trace.executed
    assert list(res.runtime_trace.loop_trip_counts.values()) == [3]
    assert "# Loop header leading trivia" in res.code
    assert "# inline loop header comment" in res.code
    assert "# [on-device unrolled iter=0/3]" in res.code
    assert "# [on-device unrolled iter=1/3]" in res.code
    assert "# [on-device unrolled iter=2/3]" in res.code
    assert "# Inside loop body comment" in res.code
    assert "# inline body op comment" in res.code

  cfg_shlo = RuntimeConfig(
    source_framework="torch",
    target_framework="stablehlo",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3)],
  )
  res_shlo = ASTEngine(semantics=sem, config=cfg_shlo).run(code)
  assert res_shlo.success
  assert "// Loop header leading trivia" in res_shlo.code
  assert "// [on-device unrolled iter=0/3]" in res_shlo.code
  assert "// [on-device unrolled iter=1/3]" in res_shlo.code
  assert "// [on-device unrolled iter=2/3]" in res_shlo.code
  assert "tensor<2x3xf32>" in res_shlo.code


def test_dynamic_for_loop_concretized_when_unroll_skipped() -> None:
  """Verify `RuntimeAnnotationTransformer.leave_For` concretizes and annotates `for` loops when unrolling does not apply."""
  # Tuple target `for i, j in range(x.size(0)):` is not unrolled by `unroll_static_loops`, so `leave_For` runs on it
  code = """import torch

def forward(x):
    for i, j in range(x.size(0)):
        x = torch.relu(x)
    return x
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="torch",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(4, 8)],
  )
  res = ASTEngine(semantics=SemanticsManager(), config=cfg).run(code)
  assert res.success
  assert "range(4" in res.code
  assert "# [on-device loop] iterations=4" in res.code


def test_range_concretizer_and_for_executor_edge_cases() -> None:
  """Cover multi-arg `range(0, x.shape[0], 2)`, early return inside `for`, and non-range iterators."""
  # 1. _concretize_range_iter with 1-arg symbolic, 2-arg symbolic, and non-range
  e1 = cst.parse_expression("range(x.shape[0])")
  assert cst.Module([]).code_for_node(_concretize_range_iter(e1, 5)) == "range(5)"

  e2 = cst.parse_expression("range(0, x.shape[0], 2)")
  assert cst.Module([]).code_for_node(_concretize_range_iter(e2, 3)) == "range(3)"

  e3 = cst.parse_expression("other_iter(x)")
  assert _concretize_range_iter(e3, 3) is e3

  e4 = cst.parse_expression("range(4)")
  assert _concretize_range_iter(e4, 4) is e4

  # 2. SourceLazyConcretizer.leave_For with empty loop_trip_counts and static range
  empty_trace = RuntimeTraceReport(executed=True, loop_trip_counts={})
  mod = cst.parse_module("""for i in range(4):
    pass
""")
  assert mod.visit(SourceLazyConcretizer(empty_trace)).code == mod.code

  trace_with_static = RuntimeTraceReport(executed=True, loop_trip_counts={1: 4})
  assert mod.visit(SourceLazyConcretizer(trace_with_static)).code == mod.code

  # 3. RuntimeAnnotationPass on module-level `for` (outside function) and exhausted loop_trip_counts
  sem = SemanticsManager()
  ctx = RewriterContext(
    semantics=sem,
    config=RuntimeConfig(
      source_framework="torch",
      target_framework="torch",
      enable_on_device_execution=True,
      inject_runtime_annotations=False,
    ),
    runtime_trace=RuntimeTraceReport(executed=True, loop_trip_counts={2: 3}),
  )
  mod_for = cst.parse_module(
    """for k in range(2):
    pass
def forward(x):
    for i in range(x.shape[0]):
        pass
    for j in range(2):
        pass
"""
  )
  out_for = RuntimeAnnotationPass().transform(mod_for, ctx).code
  assert "for i in range(3):" in out_for

  # 4. _ForwardSymbolicExecutor for-loop with 2-arg/3-arg range, negative step, non-range, and early return
  from ml_switcheroo.core.runtime_tensor import OnDeviceTraceTensor

  ex = _ForwardSymbolicExecutor({}, [OnDeviceTraceTensor(shape=(6, 4), data=[1.0] * 24)], seed=0)
  assert ex._eval_for_trip_count(cst.parse_expression("range(unknown_var)")) == 1
  assert ex._eval_for_trip_count(cst.parse_expression("range(0, 6, 2)")) == 3
  assert ex._eval_for_trip_count(cst.parse_expression("range(6, 0, -2)")) == 3
  assert ex._eval_for_trip_count(cst.parse_expression("range(0, 6, 0)")) == 1
  assert ex._eval_for_trip_count(cst.parse_expression("custom_iter(x)")) == 1

  tuple_target_mod = cst.parse_module(
    """def forward(x):
    for i, j in range(2):
        x = x + 1.0
    with ctx():
        pass
    x = x + 2.0
    return x
"""
  )
  out_tuple = ex.execute_module(tuple_target_mod)
  assert out_tuple.shape == (6, 4)

  early_ret_mod = cst.parse_module(
    """def forward(x):
    for i in range(x.shape[0]):
        return x
"""
  )
  out_t = ex.execute_module(early_ret_mod)
  assert out_t.shape == (6, 4)
