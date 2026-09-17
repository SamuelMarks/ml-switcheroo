"""Comprehensive test suite for functional state & mutation desugaring.

Covers Section 3.2 of TODO_PLAN.md:
- In-place mutation replacement:
  * Augmented assignments (x += y -> x = x + y for JAX, preserved for eager targets).
  * Indexed assignments (x[idx] = val -> x = x.at[idx].set(val) for JAX).
  * Augmented indexed assignments (x[idx] += val -> x = x.at[idx].add(val) for JAX).
  * Reverse desugaring (x = x.at[idx].set(val) -> x[idx] = val for PyTorch/MLX/Keras).
  * Standalone expression reverse desugaring (x.at[idx].set(val) -> x[idx] = val).
- PRNG key management and threading:
  * Static identification of stochastic calls and PRNGKey injection for JAX.
  * Seed conversion (PRNGKey -> manual_seed / random.seed / set_random_seed).
  * Stripping key arguments for eager frameworks.
"""

from ml_switcheroo.core.engine import ASTEngine


def test_augmented_assignment_desugaring_jax() -> None:
  """Test augmented assignments desugaring into binary operations when targeting JAX."""
  code = """def calc(x, y):
    x += y
    x -= y
    x *= y
    x /= y
    x @= y
    return x
"""
  engine = ASTEngine(source="torch", target="jax")
  res = engine.run(code)
  out = res.code

  assert "x = x + y" in out
  assert "x = x - y" in out
  assert "x = x * y" in out
  assert "x = x / y" in out
  assert "x = x @ y" in out
  assert "+=" not in out
  assert "-=" not in out


def test_augmented_assignment_preserved_eager() -> None:
  """Test augmented assignments preserved when targeting eager frameworks."""
  code = """def calc(x, y):
    x += y
    return x
"""
  engine = ASTEngine(source="torch", target="torch")
  res = engine.run(code)
  out = res.code
  assert "x += y" in out


def test_indexed_assignment_desugaring_jax() -> None:
  """Test indexed assignment x[indices] = values rewriting to x.at[indices].set(values) for JAX."""
  code = """def update_slice(x, idx, val):
    x[idx] = val
    x[1:5] = val * 2
    return x
"""
  engine = ASTEngine(source="torch", target="jax")
  res = engine.run(code)
  out = res.code

  assert "x = x.at[idx].set(val)" in out
  assert "x = x.at[1:5].set(val * 2)" in out


def test_indexed_augmented_assignment_desugaring_jax() -> None:
  """Test indexed augmented assignment x[indices] += values rewriting to x.at[indices].add(values)."""
  code = """def add_slice(x, idx, val):
    x[idx] += val
    return x
"""
  engine = ASTEngine(source="torch", target="jax")
  res = engine.run(code)
  out = res.code

  assert "x = x.at[idx].add(val)" in out


def test_reverse_desugaring_to_torch() -> None:
  """Test reverse desugaring from x.at[idx].set(val) to x[idx] = val for PyTorch."""
  code = """def update(x, idx, val):
    x = x.at[idx].set(val)
    x = x.at[idx].add(val)
    return x
"""
  engine = ASTEngine(source="jax", target="torch")
  res = engine.run(code)
  out = res.code

  assert "x[idx] = val" in out
  assert "x[idx] += val" in out
  assert ".at[" not in out


def test_reverse_desugaring_to_mlx() -> None:
  """Test reverse desugaring from x.at[idx].set(val) to x[idx] = val for Apple MLX."""
  code = """def update(x, idx, val):
    x = x.at[idx].set(val)
    return x
"""
  engine = ASTEngine(source="jax", target="mlx")
  res = engine.run(code)
  out = res.code

  assert "x[idx] = val" in out
  assert ".at[" not in out


def test_reverse_desugaring_to_keras() -> None:
  """Test reverse desugaring from x.at[idx].set(val) to x[idx] = val for Keras."""
  code = """def update(x, idx, val):
    x = x.at[idx].set(val)
    return x
"""
  engine = ASTEngine(source="jax", target="keras")
  res = engine.run(code)
  out = res.code

  assert "x[idx] = val" in out
  assert ".at[" not in out


def test_reverse_desugaring_standalone_expr() -> None:
  """Test standalone expression reverse desugaring x.at[idx].set(val) -> x[idx] = val."""
  code = """def update(x, idx, val):
    x.at[idx].set(val)
    return x
"""
  engine = ASTEngine(source="jax", target="torch")
  res = engine.run(code)
  out = res.code

  assert "x[idx] = val" in out


def test_prng_injection_for_stochastic_jax() -> None:
  """Test PRNGKey injection into functions with stochastic calls targeting JAX."""
  code = '''def sample():
    """Sample docstring."""
    x = torch.randn(10, 20)
    return x
'''
  engine = ASTEngine(source="torch", target="jax")
  res = engine.run(code)
  out = res.code

  assert "rng = jax.random.PRNGKey(0)" in out
  assert '"""Sample docstring."""' in out


def test_prng_injection_compound_first_statement() -> None:
  """Test PRNGKey injection when function starts with a compound statement without docstring."""
  code = """def sample():
    if True:
      x = torch.randn(10)
    return x
"""
  res = ASTEngine(source="torch", target="jax").run(code)
  assert "rng = jax.random.PRNGKey(0)" in res.code


def test_prng_injection_assign_first_statement() -> None:
  """Test PRNGKey injection when function starts with assignment instead of docstring."""
  code = """def sample():
    a = 1
    x = torch.randn(10)
    return x
"""
  res = ASTEngine(source="torch", target="jax").run(code)
  assert "rng = jax.random.PRNGKey(0)" in res.code


def test_prng_no_injection_if_rng_param_present() -> None:
  """Test no PRNGKey injection if function already declares an rng/key parameter."""
  code = """def sample(rng):
    x = torch.randn(10, 20)
    return x
"""
  engine = ASTEngine(source="torch", target="jax")
  res = engine.run(code)
  out = res.code

  assert "PRNGKey(0)" not in out


def test_prng_seed_conversion_to_eager() -> None:
  """Test conversion of jax.random.PRNGKey(seed) to target framework seeds."""
  code = """def init_seed():
    rng = jax.random.PRNGKey(42)
"""
  # To PyTorch
  res_torch = ASTEngine(source="jax", target="torch").run(code)
  assert "torch.manual_seed(42)" in res_torch.code

  # To MLX
  res_mlx = ASTEngine(source="jax", target="mlx").run(code)
  assert "mx.random.seed(42)" in res_mlx.code

  # To Keras
  res_keras = ASTEngine(source="jax", target="keras").run(code)
  assert "keras.utils.set_random_seed(42)" in res_keras.code

  # To Other Eager (e.g. tensorflow/numpy)
  res_other = ASTEngine(source="jax", target="numpy").run(code)
  assert "manual_seed(42)" in res_other.code or "seed(42)" in res_other.code

  # Without explicit seed argument
  code_no_args = """def init_seed():
    rng = jax.random.PRNGKey()
"""
  res_no_args = ASTEngine(source="jax", target="torch").run(code_no_args)
  assert "torch.manual_seed(0)" in res_no_args.code


def test_non_call_assign_and_expr_mutation() -> None:
  """Test that standard non-call assignments and expressions pass through unchanged."""
  code = """def simple():
    a = 1
    x = y = 1
    b, c = 2, 3
    "stand_alone_string"
    return a
"""
  res_jax = ASTEngine(source="torch", target="jax").run(code)
  assert "a = 1" in res_jax.code
  assert "x = y = 1" in res_jax.code
  res_torch = ASTEngine(source="jax", target="torch").run(code)
  assert "a = 1" in res_torch.code


def test_stochastic_key_argument_stripping() -> None:
  """Test stripping key arguments when converting from JAX to eager frameworks."""
  code = """def sample(key):
    x = jax.random.normal(key, (10, 20))
    y = jax.random.uniform((10, 20), key=key)
    return x, y
"""
  engine = ASTEngine(source="jax", target="torch")
  res = engine.run(code)
  out = res.code

  assert "jax.random.normal((10, 20))" in out
  assert "jax.random.uniform((10, 20))" in out
  assert "key=key" not in out


def test_mutation_transformer_helpers() -> None:
  """Directly test guard branches in FunctionalMutationTransformer."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.passes.mutation import FunctionalMutationTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.config import RuntimeConfig

  mgr = SemanticsManager()
  cfg = RuntimeConfig.load(source="torch", target="torch")
  ctx = RewriterContext(semantics=mgr, config=cfg)
  transformer = FunctionalMutationTransformer(ctx)

  # _match_at_call guard branches
  # 1. Non-attribute func
  call1 = cst.Call(func=cst.Name("foo"))
  assert transformer._match_at_call(call1) is None

  # 2. Method not in allowed set
  call2 = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("bar")), args=[cst.Arg(value=cst.Name("y"))])
  assert transformer._match_at_call(call2) is None

  # 3. No args
  call3 = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("set")))
  assert transformer._match_at_call(call3) is None

  # 4. func.value is not a Subscript
  call4 = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("set")),
    args=[cst.Arg(value=cst.Name("y"))],
  )
  assert transformer._match_at_call(call4) is None

  # 5. Subscript value is not an Attribute with 'at'
  sub5 = cst.Subscript(value=cst.Name("x"), slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Name("i")))])
  call5 = cst.Call(
    func=cst.Attribute(value=sub5, attr=cst.Name("set")),
    args=[cst.Arg(value=cst.Name("y"))],
  )
  assert transformer._match_at_call(call5) is None

  # 6. Nested attribute func name extraction in leave_Call
  call6 = cst.Call(
    func=cst.Attribute(
      value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nested")),
      attr=cst.Name("randn"),
    )
  )
  res_call6 = transformer.leave_Call(call6, call6)
  assert isinstance(res_call6, cst.Call)
  assert transformer._stochastic_call_count > 0

  call_simple = cst.Call(func=cst.Name("randn"))
  res_simple = transformer.leave_Call(call_simple, call_simple)
  assert isinstance(res_simple, cst.Call)

  # Non-matching Call on leave_Call (func is Lambda, neither Attribute nor Name)
  call_lambda = cst.Call(func=cst.Lambda(params=cst.Parameters(), body=cst.Name("x")))
  res_lambda = transformer.leave_Call(call_lambda, call_lambda)
  assert res_lambda.deep_equals(call_lambda)

  # Non-matching Call on leave_Call (func is 1-level Attribute)
  call_1level = cst.Call(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("abs")))
  res_1level = transformer.leave_Call(call_1level, call_1level)
  assert res_1level.deep_equals(call_1level)

  # Non-matching Call on leave_Call (func is complex Attribute with non-Name value)
  call_other = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("get_fn")), attr=cst.Name("bar")))
  res_other = transformer.leave_Call(call_other, call_other)
  assert res_other.deep_equals(call_other)

  # Non-matching Call on leave_Expr when target is eager (not _match_at_call)
  expr_call_no_match = cst.Expr(value=cst.Call(func=cst.Name("foo")))
  res_expr = transformer.leave_Expr(expr_call_no_match, expr_call_no_match)
  assert res_expr.deep_equals(expr_call_no_match)

  # Non-matching Call on leave_Assign when target is eager (not _match_at_call)
  assign_call_no_match = cst.Assign(
    targets=[cst.AssignTarget(target=cst.Name("x"))],
    value=cst.Call(func=cst.Name("foo")),
  )
  res_assign = transformer.leave_Assign(assign_call_no_match, assign_call_no_match)
  assert res_assign.deep_equals(assign_call_no_match)

  # 7. Unmapped aug assign operator
  aug7 = cst.AugAssign(
    target=cst.Name("x"),
    operator=cst.Add(),  # type: ignore
    value=cst.Name("y"),
  )
  ctx_jax = RewriterContext(semantics=mgr, config=RuntimeConfig.load(source="torch", target="jax"))
  trans_jax = FunctionalMutationTransformer(ctx_jax)
  assert trans_jax.leave_AugAssign(aug7, aug7) == aug7

  # Test FunctionalMutationPass entrypoint
  from ml_switcheroo.core.rewriter.passes.mutation import FunctionalMutationPass

  mutation_pass = FunctionalMutationPass()
  mod = cst.parse_module("x += 1")
  transformed_mod = mutation_pass.transform(mod, ctx_jax)
  assert transformed_mod is not None

  # Branch 192->204: leave_FunctionDef when function body is SimpleStatementSuite (not IndentedBlock)
  trans_jax._stochastic_call_count = 1
  trans_jax._has_rng_param = False
  fn_simple: cst.FunctionDef = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  res_fn = trans_jax.leave_FunctionDef(fn_simple, fn_simple)
  assert res_fn is fn_simple

  # Branch 230->248: leave_AugAssign when subscript target value is not BaseAssignTargetExpression
  aug_non_assign_target = cst.AugAssign(
    target=cst.Subscript(
      value=cst.Call(func=cst.Name("get_arr")),
      slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))],
    ),
    operator=cst.AddAssign(),
    value=cst.Integer("1"),
  )
  res_aug2 = trans_jax.leave_AugAssign(aug_non_assign_target, aug_non_assign_target)
  assert isinstance(res_aug2, cst.Assign)

  # Branch 274->293: leave_Assign when target subscript value is not BaseAssignTargetExpression
  assign_non_assign_target = cst.Assign(
    targets=[
      cst.AssignTarget(
        target=cst.Subscript(
          value=cst.Call(func=cst.Name("get_arr")),
          slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))],
        )
      )
    ],
    value=cst.Integer("1"),
  )
  res_assign2 = trans_jax.leave_Assign(assign_non_assign_target, assign_non_assign_target)
  assert res_assign2 is assign_non_assign_target

  # Branch 298->312: leave_Assign (eager target) when base_expr is not BaseAssignTargetExpression
  ctx_torch = RewriterContext(semantics=mgr, config=RuntimeConfig.load(source="jax", target="torch"))
  trans_torch = FunctionalMutationTransformer(ctx_torch)
  assign_at_call = cst.Assign(
    targets=[cst.AssignTarget(target=cst.Name("x"))],
    value=getattr(cst.parse_statement("get_arr().at[0].set(1)"), "body")[0].value,  # type: ignore[attr-defined]
  )
  res_assign3 = trans_torch.leave_Assign(assign_at_call, assign_at_call)
  assert res_assign3 is assign_at_call

  # Branch 329->335: leave_Expr (eager target) when base_expr is not BaseAssignTargetExpression
  expr_at_call = cst.Expr(
    value=getattr(cst.parse_statement("get_arr().at[0].set(1)"), "body")[0].value  # type: ignore[attr-defined]
  )
  res_expr2 = trans_torch.leave_Expr(expr_at_call, expr_at_call)
  assert res_expr2 is expr_at_call
