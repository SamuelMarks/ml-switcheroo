"""Unit tests for Phase 17: Interweaving Dynamic Symbolic Arguments (`reshape(-1)`, `view(-1)`, `x.shape[i]`, `SymInt`) in CST Rewriting.

Verifies:
- `x.view(x.size(0), -1)` and `x.reshape(x.shape[0], -1)` concretize to `(2, 48)` when input is `(2, 3, 4, 4)`.
- `jnp.reshape(x, (B, -1))` and `reshape(x, (-1, 10))` concretize tuple/list and variadic symbolic arguments while preserving CST trivia.
- `nn.Linear(x.shape[-1], 10)(x)` and `nn.LayerNorm(x.shape[-1])(x)` concretize symbolic constructor arguments.
- All helper branches in `_is_symbolic_dim_expr`, `_concretize_reshape_view_call`, and `_concretize_symbolic_ctor_expr`.
"""

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.rewriter.passes.runtime_annotations import (
  _concretize_reshape_view_call,
  _concretize_symbolic_ctor_expr,
  _is_symbolic_dim_expr,
)
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry
from ml_switcheroo.semantics.manager import SemanticsManager


def test_symbolic_reshape_and_view_concretization_end_to_end() -> None:
  """Verify x.view(x.size(0), -1) and x.reshape(x.shape[0], -1) concretize to (2, 48) with trivia preserved."""
  code = """import torch
import torch.nn as nn

class SymbolicReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        # Leading comment on dynamic view
        B = x.size(0)  # inline batch size comment
        flat = x.view(x.size(0), -1)  # inline view comment
        reshaped = flat.reshape(x.shape[0], -1)  # inline reshape comment
        out = self.fc(reshaped)
        return out.reshape(B, -1)  # inline return reshape comment
"""
  sem = SemanticsManager()
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  res = ASTEngine(semantics=sem, config=cfg).run(code)
  assert res.success
  assert res.runtime_trace is not None
  assert res.runtime_trace.executed
  # Concrete dimensions (2, 48) and (2, 10) interwoven into reshape/view calls
  assert "2, 48" in res.code
  assert "2, 10" in res.code
  assert "in_features=48" in res.code
  # All leading and inline trivia preserved
  assert "# Leading comment on dynamic view" in res.code
  assert "# inline batch size comment" in res.code
  assert "# inline view comment" in res.code
  assert "# inline reshape comment" in res.code
  assert "# inline return reshape comment" in res.code


def test_inline_symbolic_constructor_concretization() -> None:
  """Verify nn.Linear(x.shape[-1], 10)(x) and nn.LayerNorm(x.shape[-1])(x) concretize in-place."""
  code = """import torch.nn as nn

def forward(x):
    # Inline LayerNorm with symbolic x.shape[-1]
    normed = nn.LayerNorm(x.shape[-1])(x)
    # Inline Linear with symbolic x.shape[-1]
    out = nn.Linear(x.shape[-1], 10)(normed)
    # Inline LazyLinear
    lazy_out = nn.LazyLinear(6)(out)
    return lazy_out
"""
  sem = SemanticsManager()
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="torch",
    enable_on_device_execution=True,
    input_shapes=[(2, 24)],
  )
  res = ASTEngine(semantics=sem, config=cfg).run(code)
  assert res.success
  assert "LayerNorm(24" in res.code
  assert "Linear(24, 10)" in res.code
  assert "# Inline LayerNorm with symbolic x.shape[-1]" in res.code


def test_concretize_symbolic_helpers_unit_branches() -> None:
  """Verify all branches of _is_symbolic_dim_expr, _concretize_reshape_view_call, and _concretize_symbolic_ctor_expr."""
  assert _is_symbolic_dim_expr(cst.parse_expression("-1")) is True
  assert _is_symbolic_dim_expr(cst.parse_expression("x.shape[0]")) is True
  assert _is_symbolic_dim_expr(cst.parse_expression("x.size(0)")) is True
  assert _is_symbolic_dim_expr(cst.parse_expression("B")) is True
  assert _is_symbolic_dim_expr(cst.parse_expression("16")) is False

  # Non-call or empty shape returns unchanged
  non_call = cst.parse_expression("x + 1")
  assert _concretize_reshape_view_call(non_call, [2, 8]) is non_call
  call_expr = cst.parse_expression("x.reshape(-1, 8)")
  assert _concretize_reshape_view_call(call_expr, []) is call_expr

  # Non-reshape/view call returns unchanged
  other_call = cst.parse_expression("x.sum(0)")
  assert _concretize_reshape_view_call(other_call, [2]) is other_call

  # Bare function `reshape(x, (B, -1))`
  bare_reshape = cst.parse_expression("reshape(x, (B, -1))")
  concretized_bare = _concretize_reshape_view_call(bare_reshape, [4, 16])
  assert cst.Module([]).code_for_node(concretized_bare) == "reshape(x, (4, 16))"

  # Module function with variadic args `torch.reshape(x, B, -1)`
  var_reshape = cst.parse_expression("torch.reshape(x, B, -1)")
  concretized_var = _concretize_reshape_view_call(var_reshape, [4, 16])
  assert cst.Module([]).code_for_node(concretized_var) == "torch.reshape(x, 4, 16)"

  # Already concrete tuple returns unchanged
  concrete_tuple = cst.parse_expression("jnp.reshape(x, (4, 16))")
  assert _concretize_reshape_view_call(concrete_tuple, [4, 16]) is concrete_tuple

  # Mismatched rank returns unchanged
  mismatch = cst.parse_expression("x.view(-1)")
  assert _concretize_reshape_view_call(mismatch, [4, 16]) is mismatch

  # _concretize_symbolic_ctor_expr branches
  tel_empty = NodeRuntimeTelemetry(node_id="n0", var_name="y", op_type="Linear", input_shapes=[], output_shape=[])
  assert _concretize_symbolic_ctor_expr(non_call, tel_empty) is non_call
  assert _concretize_symbolic_ctor_expr(cst.parse_expression("Linear(in_dim, 10)"), tel_empty) is not None

  tel_out_only = NodeRuntimeTelemetry(node_id="n1", var_name="y", op_type="Linear", input_shapes=[], output_shape=[2, 32])
  direct_ctor = cst.parse_expression("Linear(in_dim, 10)")
  updated_direct = _concretize_symbolic_ctor_expr(direct_ctor, tel_out_only)
  assert cst.Module([]).code_for_node(updated_direct) == "Linear(32, 10)"

  attr_ctor = cst.parse_expression("nn.Dense(in_dim)")
  updated_attr = _concretize_symbolic_ctor_expr(attr_ctor, tel_out_only)
  assert cst.Module([]).code_for_node(updated_attr) == "nn.Dense(32)"

  # Direct call that is not a constructor returns unchanged
  non_ctor_call = cst.parse_expression("foo.bar(10)")
  assert _concretize_symbolic_ctor_expr(non_ctor_call, tel_out_only) is non_ctor_call

  # Chained call where first arg is already concrete integer returns unchanged
  concrete_chained = cst.parse_expression("nn.Linear(32, 10)(x)")
  assert _concretize_symbolic_ctor_expr(concrete_chained, tel_out_only) is concrete_chained

  # Trace execution with out-of-bounds shape[99], size(), size(99), and non-scalar dim expr
  edge_code = """import torch.nn as nn

def forward(x):
    s0 = x.size()
    s_oob = x.size(99)
    sh_oob = x.shape[99]
    l1 = nn.Linear(x, x)(x)
    l2 = nn.LazyLinear(x)(x)
    return x.reshape(x, -1)
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="torch",
    enable_on_device_execution=True,
    inject_runtime_annotations=False,
    input_shapes=[(2, 8)],
  )
  res_edge = ASTEngine(semantics=SemanticsManager(), config=cfg).run(edge_code)
  assert res_edge.success
