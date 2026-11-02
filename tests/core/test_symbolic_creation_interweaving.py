"""Unit tests for Phase 22: Symbolic Factory & Slice Call Interweaving (`zeros`, `ones`, `full`, `empty`, `randn`, `rand`, `arange`, `linspace`).

Verifies:
- `torch.zeros(x.size(0), x.size(1), dtype=...)`, `jnp.ones((x.shape[0], -1))`, `torch.full((x.size(0), x.size(1)), 0.5)`, `torch.empty(x.shape[0], 4)`, `torch.randn(x.size(0), x.size(1))`, `torch.arange(x.shape[-1])`, and `torch.linspace(0, 1, steps=x.shape[-1])` concretize symbolic dimension arguments with runtime output shapes while preserving keyword arguments and CST trivia.
- 100% line, branch, and function coverage of `_concretize_creation_call` and `_eval_creation_call`.
"""

from __future__ import annotations

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.rewriter.passes.runtime_annotations import _concretize_creation_call
from ml_switcheroo.core.runtime_tensor import OnDeviceTraceTensor
from ml_switcheroo.core.runtime_tracer import _ForwardSymbolicExecutor
from ml_switcheroo.semantics.manager import SemanticsManager


def test_symbolic_creation_calls_end_to_end_with_trivia() -> None:
  """Verify symbolic `zeros`, `ones`, `full`, `randn`, `arange`, and `linspace` calls concretize and preserve trivia."""
  code = """import torch

def forward(x):
    # Leading comment on zeros creation
    z = torch.zeros(x.size(0), x.size(1), dtype=torch.float32)  # inline zeros comment
    o = torch.ones((x.shape[0], -1))
    f = torch.full((x.size(0), x.size(1)), 2.5)
    r = torch.randn(x.size(0), x.size(1))
    a = torch.arange(x.shape[-1])
    l = torch.linspace(0.0, 1.0, steps=x.shape[-1])
    return z + o + f + r
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="torch",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 8)],
  )
  res = ASTEngine(semantics=SemanticsManager(), config=cfg).run(code)
  assert res.success
  assert "# Leading comment on zeros creation" in res.code
  assert "# inline zeros comment" in res.code
  assert "torch.zeros(2, 8" in res.code
  assert "torch.ones((2, 8)" in res.code
  assert "torch.full((2, 8), 2.5" in res.code
  assert "torch.randn(2, 8" in res.code
  assert "torch.arange(8" in res.code
  assert "steps=8" in res.code


def test_concretize_creation_call_and_executor_branches() -> None:
  """Cover all branches of `_concretize_creation_call` and `_ForwardSymbolicExecutor._eval_creation_call`."""
  # 1. Non-call or empty output_shape or non-creation op
  name_expr = cst.parse_expression("x")
  assert _concretize_creation_call(name_expr, [2, 4]) is name_expr
  call_non_creation = cst.parse_expression("torch.relu(x)")
  assert _concretize_creation_call(call_non_creation, [2, 4]) is call_non_creation

  # 2. Bare function call `zeros(B, D)` and static `zeros((2, 4))`
  bare_zeros = cst.parse_expression("zeros(B, D)")
  assert cst.Module([]).code_for_node(_concretize_creation_call(bare_zeros, [2, 4])) == "zeros(2, 4)"

  static_tuple_zeros = cst.parse_expression("torch.zeros((2, 4))")
  assert _concretize_creation_call(static_tuple_zeros, [2, 4]) is static_tuple_zeros

  static_var_zeros = cst.parse_expression("torch.zeros(2, 4)")
  assert _concretize_creation_call(static_var_zeros, [2, 4]) is static_var_zeros

  # 3. `arange` 1-arg static, 2-arg symbolic, 2-arg static, and 0-arg
  arange_static = cst.parse_expression("torch.arange(8)")
  assert _concretize_creation_call(arange_static, [8]) is arange_static

  arange_2arg_sym = cst.parse_expression("torch.arange(2, x.shape[1])")
  assert cst.Module([]).code_for_node(_concretize_creation_call(arange_2arg_sym, [6])) == "torch.arange(2, 8)"

  arange_2arg_sym_start = cst.parse_expression("torch.arange(start_n, x.shape[1])")
  assert cst.Module([]).code_for_node(_concretize_creation_call(arange_2arg_sym_start, [6])) == "torch.arange(start_n, 6)"

  arange_2arg_static = cst.parse_expression("torch.arange(2, 8)")
  assert _concretize_creation_call(arange_2arg_static, [6]) is arange_2arg_static

  arange_0arg = cst.parse_expression("torch.arange()")
  assert _concretize_creation_call(arange_0arg, [6]) is arange_0arg

  # 4. `linspace` positional 3rd arg symbolic & static
  linspace_pos = cst.parse_expression("jnp.linspace(0.0, 1.0, x.shape[-1])")
  assert cst.Module([]).code_for_node(_concretize_creation_call(linspace_pos, [12])) == "jnp.linspace(0.0, 1.0, 12)"

  linspace_static = cst.parse_expression("jnp.linspace(0.0, 1.0, 12)")
  assert _concretize_creation_call(linspace_static, [12]) is linspace_static

  linspace_2arg = cst.parse_expression("jnp.linspace(0.0, 1.0)")
  assert _concretize_creation_call(linspace_2arg, [12]) is linspace_2arg

  # 5. Keyword `shape=N` scalar symbolic and static
  kw_shape_sym = cst.parse_expression("jnp.Empty(shape=x.shape[0])")
  assert cst.Module([]).code_for_node(_concretize_creation_call(kw_shape_sym, [5])) == "jnp.Empty(shape=5)"

  kw_shape_static = cst.parse_expression("jnp.empty(shape=5)")
  assert _concretize_creation_call(kw_shape_static, [5]) is kw_shape_static

  full_1d_sym = cst.parse_expression("torch.full(x.size(0), 2.5)")
  assert cst.Module([]).code_for_node(_concretize_creation_call(full_1d_sym, [5])) == "torch.full(5, 2.5)"

  # 6. `_ForwardSymbolicExecutor._eval_creation_call` branches
  ex = _ForwardSymbolicExecutor({}, [OnDeviceTraceTensor(shape=(3, 5), data=[1.0] * 15)], seed=0)
  for expr_str, expected_shape in [
    ("torch.arange(0, 6, 2)", (3,)),
    ("torch.arange()", (5,)),
    ("torch.linspace(0.0, 1.0, 7)", (7,)),
    ("torch.linspace(0.0, 1.0)", (5,)),
    ("jnp.ones(shape=(3, -1))", (3, 5)),
    ("torch.empty(3, 5)", (3, 5)),
    ("torch.full((3, 5), fill_value=3.0)", (3, 5)),
    ("torch.full((3, 5))", (3, 5)),
    ("torch.zeros()", (3, 5)),
    ("jax.numpy.relu(x)", (3, 5)),
  ]:
    t, _ = ex._eval_expr(cst.parse_expression(expr_str))
    assert t.shape == expected_shape
