"""Unit and integration tests for `src/ml_switcheroo/core/runtime_tracer.py`."""

from __future__ import annotations
from tests.conftest import _test_setattr

import sys
import types
from unittest import mock

import libcst as cst

from ml_switcheroo.config import ExecutionBackend, LossReduction, RuntimeConfig
from ml_switcheroo.core.runtime_models import ConstructorDimResolution
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _conv_nd,
  _generate_seeded_values,
  _matmul_2d,
  _prod,
)
from ml_switcheroo.core.runtime_tracer import (
  DeclaredLayerSpec,
  _ForwardSymbolicExecutor,
  _TracedLayerInstance,
  _check_all_finite,
  _compute_loss,
  _execute_autodiff_strategy,
  _extract_dims_from_annotation_text,
  _extract_int_from_expr,
  _get_call_leaf_name,
  _resolve_traits,
  infer_input_shapes_from_cst,
  run_on_device_trace,
  zero_framework_sandbox,
)
from ml_switcheroo.semantics.manager import SemanticsManager


def test_resolve_traits_adapter_and_semantics_fallback() -> None:
  """Verify _resolve_traits resolves from adapter, semantics config, or empty default."""
  t_torch = _resolve_traits("torch")
  assert t_torch.zero_runtime_package == "zero_torch"

  import ml_switcheroo.semantics.manager as mgr_mod

  prev = mgr_mod._MANAGER_CACHE
  mgr_mod._MANAGER_CACHE = {}
  sem = SemanticsManager()
  mgr_mod._MANAGER_CACHE = prev
  sem.framework_configs = {"custom_fw": {"traits": {"zero_runtime_package": "zero_custom"}}}
  t_custom = _resolve_traits("custom_fw", sem)
  assert t_custom.zero_runtime_package == "zero_custom"

  sem.framework_configs["invalid_traits_fw"] = {"traits": "not_a_dict"}
  t_empty = _resolve_traits("invalid_traits_fw", sem)
  assert t_empty.zero_runtime_package is None

  t_missing = _resolve_traits("nonexistent_fw_xyz", None)
  assert t_missing.zero_runtime_package is None

  mock_adapter = mock.MagicMock()
  mock_adapter.structural_traits = "not_structural_traits"
  with mock.patch("ml_switcheroo.core.runtime_tracer.get_adapter", return_value=mock_adapter):
    t_bad_adapter = _resolve_traits("bad_adapter")
    assert t_bad_adapter.zero_runtime_package is None


def test_zero_framework_sandbox_aliasing_and_restore() -> None:
  """Verify zero_framework_sandbox aliases importable zero-* modules and restores sys.modules."""
  fake_zero_torch = types.ModuleType("zero_torch")
  fake_zero_chex = types.ModuleType("zero_chex")
  sentinel_existing = types.ModuleType("existing_chex")

  def fake_import(name: str) -> types.ModuleType:
    """Import stub returning fake zero_torch and zero_chex modules.

    Args:
        name: Module path to import.

    Returns:
        Fake module or raises ImportError.
    """
    if name == "zero_torch":
      return fake_zero_torch
    if name == "zero_chex":
      return fake_zero_chex
    raise ImportError(name)

  sys.modules["chex"] = sentinel_existing
  sys.modules.pop("torch", None)
  try:
    with mock.patch("importlib.import_module", side_effect=fake_import):
      with zero_framework_sandbox("torch", "jax") as bindings:
        assert bindings["torch"] is fake_zero_torch
        assert bindings["chex"] is fake_zero_chex
        assert sys.modules["torch"] is fake_zero_torch
        assert sys.modules["chex"] is fake_zero_chex
      assert "torch" not in sys.modules
      assert sys.modules["chex"] is sentinel_existing
  finally:
    sys.modules.pop("chex", None)


def test_ast_helpers_and_infer_input_shapes_from_cst() -> None:
  """Verify _extract_int_from_expr, _get_call_leaf_name, and infer_input_shapes_from_cst across layer types."""
  assert _extract_int_from_expr(cst.Integer("16")) == 16
  assert _extract_int_from_expr(cst.Float("1.5")) is None
  bad_int = cst.Integer("16")
  with mock.patch.object(cst.Integer, "value", new_callable=mock.PropertyMock, return_value="not_int"):
    assert _extract_int_from_expr(bad_int) is None

  assert _get_call_leaf_name(cst.Name("Linear")) == "Linear"
  assert _get_call_leaf_name(cst.parse_expression("nn.Conv2d")) == "Conv2d"
  assert _get_call_leaf_name(cst.parse_expression("fn()()")) == ""

  assert _extract_dims_from_annotation_text("Float[Array, '2 16']") == (2, 16)
  assert _extract_dims_from_annotation_text("Tensor[99999]") is None
  assert _extract_dims_from_annotation_text("Tensor[4]") is None

  # 1. Linear positional + non-int kwarg (e.g. bias=True)
  tree_lin = cst.parse_module(
    """class M:
  def __init__(self):
    self.fc = nn.Linear(12, 6, bias=True)
  def helper(self, z):
    return z
  def forward(self, x):
    return self.fc(x)
"""
  )
  shapes, dtype, layers = infer_input_shapes_from_cst(tree_lin)
  assert shapes == [(2, 12)]
  assert dtype == "float32"
  assert layers["fc"].in_features == 12
  assert layers["fc"].out_features == 6
  assert layers["fc"].is_lazy is False
  rep_lin = run_on_device_trace(
    """import torch
class HelperNoForward:
  def not_forward(self, x):
    return x
class M:
  def __init__(self):
    self.fc = nn.Linear(12, 12, bias=True)
    self.dead = nn.Linear(12, 4)
  def helper(self, z):
    return z
  def forward(self, x):
    _unused = self.dead(x)
    for _i in []:
      pass
    if x.sum() < -999.0:
      x = x
    else:
      x = x + 0.1
    if x.sum() < -999.0:
      x = x
    elif x.sum() > -999.0:
      x = x + 0.1
    while x.sum() > -999.0:
      x = x + 0.1
    h = self.fc(x)
    return self.fc(h)
def extra_fn(x):
  return x
""",
    "torch",
    "jax",
  )
  assert rep_lin.executed is True

  # 2. Linear single positional (lazy / flax style) and Linear keyword without in_features
  tree_lin_lazy = cst.parse_module(
    """class M:
  def __init__(self):
    self.fc1 = Linear(10)
    self.fc2 = Linear(out_features=5)
"""
  )
  shapes_lz, _, layers_lz = infer_input_shapes_from_cst(tree_lin_lazy)
  assert shapes_lz == [(2, 4)]
  assert layers_lz["fc1"].is_lazy is True
  assert layers_lz["fc1"].out_features == 10
  assert layers_lz["fc2"].is_lazy is True

  # 3. LazyLinear & Dense
  tree_dense = cst.parse_module(
    """class M:
  def setup(self):
    self.l1 = nn.LazyLinear(8)
    self.l2 = layers.Dense(units=4)
"""
  )
  _, _, layers_d = infer_input_shapes_from_cst(tree_dense)
  assert layers_d["l1"].is_lazy is True
  assert layers_d["l1"].out_features == 8
  assert layers_d["l2"].is_lazy is True
  assert layers_d["l2"].out_features == 4

  # 4. Conv1d, Conv2d, Conv3d, and lazy Conv
  tree_c1 = cst.parse_module(
    """class M:
  def __init__(self):
    self.c = nn.Conv1d(3, 6, 3)
"""
  )
  assert infer_input_shapes_from_cst(tree_c1)[0] == [(2, 3, 8)]

  tree_c2 = cst.parse_module(
    """class M:
  def __init__(self):
    self.c = nn.Conv2d(3, 8, 3)
"""
  )
  assert infer_input_shapes_from_cst(tree_c2)[0] == [(2, 3, 8, 8)]

  tree_c3 = cst.parse_module(
    """class M:
  def __init__(self):
    self.c = nn.Conv3d(in_channels=2, out_channels=4)
"""
  )
  assert infer_input_shapes_from_cst(tree_c3)[0] == [(2, 2, 4, 4, 4)]

  tree_c_lazy = cst.parse_module(
    """class M:
  def __init__(self):
    self.c = Conv2D(filters=8)
"""
  )
  assert infer_input_shapes_from_cst(tree_c_lazy)[2]["c"].is_lazy is True

  # 5. Embedding & MultiheadAttention
  tree_emb = cst.parse_module(
    """class M:
  def __init__(self):
    self.e = nn.Embedding(100, 16)
"""
  )
  s_emb, dt_emb, _ = infer_input_shapes_from_cst(tree_emb)
  assert s_emb == [(2, 8)]
  assert dt_emb == "int32"

  tree_mha = cst.parse_module(
    """class M:
  def __init__(self):
    self.attn = nn.MultiheadAttention(32, 4)
"""
  )
  assert infer_input_shapes_from_cst(tree_mha)[0] == [(2, 4, 32)]

  # 6. Praxis create_child and non-self assign / non-init assign / empty call leaf
  tree_praxis = cst.parse_module(
    """class M:
  def setup(self):
    other.x = nn.Linear(4, 4)
    local_v = 1
    self.bad = get_fn()(4)
    self.create_child('fc', Linear(input_dims=6, output_dims=3))
  def __call__(self, x: Float[Array, '4 6']):
    self.outside = nn.Linear(2, 2)
    return x
"""
  )
  s_px, _, l_px = infer_input_shapes_from_cst(tree_praxis)
  assert s_px == [(2, 6)]
  assert "fc" in l_px

  # 7. Annotation inference when no layers in __init__ (y: int first, then x: Tensor[2, 10])
  tree_fn = cst.parse_module(
    """def forward(self, y: int, x: Tensor[2, 10]):
  return x
"""
  )
  assert infer_input_shapes_from_cst(tree_fn)[0] == [(2, 10)]

  # 8. Config override
  cfg = RuntimeConfig(input_shapes=[(4, 16)], input_dtype="float16")
  s_ovr, dt_ovr, _ = infer_input_shapes_from_cst(tree_fn, cfg)
  assert s_ovr == [(4, 16)]
  assert dt_ovr == "float16"


def test_on_device_trace_tensor_operations_and_autodiff() -> None:
  """Verify OnDeviceTraceTensor constructors, ops, broadcasting, masking, and backward VJPs."""
  assert _prod((2, 3)) == 6
  assert len(_generate_seeded_values(4, 42, "int32")) == 4
  assert len(_generate_seeded_values(20, 1, "float32", scale=0.01)) == 20

  # Default data, truncation, and padding branches
  t_default = OnDeviceTraceTensor((2, 2))
  assert len(t_default.data) == 4
  t_trunc = OnDeviceTraceTensor((2,), data=[1.0, 2.0, 3.0])
  assert t_trunc.data == [1.0, 2.0]
  t_pad = OnDeviceTraceTensor((3,), data=[2.0])
  assert t_pad.data == [2.0, 2.0, 2.0]
  t_empty_pad = OnDeviceTraceTensor((2,), data=[])
  assert t_empty_pad.data == [0.0, 0.0]
  t_empty_item = OnDeviceTraceTensor((1,), data=[5.0])
  t_empty_item.data = []
  assert t_empty_item.item() == 0.0
  assert t_default.ndim == 2

  # Arithmetic, broadcasting (both directions), and activations
  a = OnDeviceTraceTensor((2, 2), data=[1.0, -2.0, 3.0, 4.0])
  b = OnDeviceTraceTensor((2,), data=[2.0, 0.5])
  c = (a + b) - (b + a) + (2.0 + a) * (3.0 * b) / 2.0
  d = c.relu().sigmoid().tanh().exp()
  flat_1d = OnDeviceTraceTensor((3,), data=[1.0, 2.0, 3.0]).flatten()
  assert flat_1d.shape == (3,)
  flat_2d = a.flatten(1)
  assert flat_2d.shape == (2, 2)
  r_static = a.reshape(4)
  assert r_static.shape == (4,)
  r_empty = a.reshape()
  assert r_empty.shape == (2, 2)
  r_dyn = a.view([-1, 2])
  assert r_dyn.is_dynamic is True
  tr_1d = flat_1d.transpose()
  assert tr_1d.shape == (3,)
  tr_2d = a.transpose(0, 1)
  assert tr_2d.shape == (2, 2)

  # Comparisons, bool, and indexing (boolean mask with matches, all-false mask, slice index)
  mask_gt = a > 0.0
  mask_lt = a < OnDeviceTraceTensor((), data=[-10.0])
  assert bool(mask_gt) is True
  assert bool(mask_lt) is False
  assert bool(a) is True

  masked_a = a[mask_gt]
  assert masked_a.is_dynamic is True
  masked_empty = a[mask_lt]
  assert masked_empty.shape == (1,)
  sliced_a = a[0]
  assert sliced_a.shape == a.shape

  loss = d.sum() + masked_a.mean() + sliced_a.sum() + tr_2d.sum()
  # Add a node that does not require grad to test requires_grad=False branch in backward
  no_grad_t = OnDeviceTraceTensor((2, 2), data=[1.0, 1.0, 1.0, 1.0], requires_grad=False)
  total = loss + (a / b).sum() + (b / a).sum() + no_grad_t.sum()
  total.backward()
  assert a.grad is not None
  assert b.grad is not None


def test_matmul_and_conv_and_layer_instances() -> None:
  """Verify _matmul_2d (1D and 2D, with and without bias), _conv_nd, and _TracedLayerInstance."""
  x1 = OnDeviceTraceTensor((4,), data=[1.0, 2.0, 3.0, 4.0])
  w1 = OnDeviceTraceTensor((2, 4), data=[0.5] * 8)
  out1 = _matmul_2d(x1, w1, None)
  out1.sum().backward()
  assert out1.shape == (2,)
  assert x1.grad is not None

  x_conv = OnDeviceTraceTensor((2, 2, 4, 4))
  w_conv = OnDeviceTraceTensor((3, 2, 3, 3))
  out_conv_no_bias = _conv_nd(x_conv, w_conv, None, 3)
  out_conv_no_bias.sum().backward()
  assert out_conv_no_bias.shape == (2, 3, 4, 4)

  resolved: dict[str, ConstructorDimResolution] = {}
  params: dict[str, OnDeviceTraceTensor] = {}
  for op_name in ("Flatten", "ReLU", "Sigmoid", "Tanh", "Conv1d", "Conv2d"):
    spec = DeclaredLayerSpec(f"l_{op_name}", op_name, None, 3, True)
    inst = _TracedLayerInstance(spec, 42, resolved, params)
    inp = OnDeviceTraceTensor((2, 2, 4, 4)) if "Conv" in op_name else OnDeviceTraceTensor((2, 4))
    res = inst(inp)
    # Call second time to hit cached parameter branch
    res2 = inst(inp)
    res2.sum().backward()
    assert res.shape == res2.shape


def test_symbolic_executor_and_autodiff_strategies() -> None:
  """Verify _ForwardSymbolicExecutor across expressions, branches, loops, and autodiff strategies."""
  code = """
class DynamicModel(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 4, 3)
        self.flat = nn.Flatten()
        self.fc = nn.LazyLinear(2)

    def forward(self, x, extra):
        h = self.conv(x)
        h = self.flat(h)
        h = h.flatten(1)
        h = h.reshape(-1, 256)
        h = h.reshape((2, -1))
        h = h.relu().sigmoid().tanh()
        h = h.transpose()
        s = h.sum() + h.mean()
        h = self.undeclared_layer(h)
        h = -h + 1 - 0.5 * h / 2.0
        h = h @ h
        h += 1.0
        F.relu(h)
        F.sigmoid(h)
        F.tanh(h)
        F.exp(h)
        F.flatten(h)
        F.sum(h)
        F.mean(h)
        F.reshape(h, -1, 2)
        F.reshape(h, (2, 2))
        F.add(h, h)
        F.mul(h, h)
        F.matmul(h, h)
        F.nonzero(h)
        F.custom_op(h)
        self.attr_target = h
        a, b = h, h
        idx_slice = h[0:1]
        idx_int = h[0]
        if h.sum() > -999.0:
            h = self.fc(h)
        else:
            h = h
        if h.sum() < -999.0:
            h = h
        elif h.sum() > -999.0:
            h = h * 1.0
        if h.sum() < -999.0:
            h = h
        else:
            h = h + 1.0
        while h.sum() > -999.0:
            h = h + 0.1
        mask = h > 0
        z = h[mask]
        return z
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    execution_backend=ExecutionBackend.AUTO,
    loss_reduction=LossReduction.L2,
  )
  report = run_on_device_trace(code, "torch", "flax_nnx", config=cfg)
  assert report.executed is True
  assert "fc" in report.resolved_constructors
  assert len(report.resolved_branches) >= 3
  assert len(report.loop_trip_counts) == 1
  assert report.loss_value is not None
  assert report.total_grad_norm is not None
  assert report.forward_kernel_code is not None
  assert report.backward_kernel_code is not None

  # Test early return inside if / else / elif / while and empty module / raw function
  code_early_returns = """
def fn1(self):
    if x > -999:
        return x
"""
  assert run_on_device_trace(code_early_returns, "jax", "torch").executed is True

  code_else_return = """
def fn2(x):
    if x < -999:
        y = x
    else:
        return x
"""
  assert run_on_device_trace(code_else_return, "jax", "torch").executed is True

  code_elif_return = """
def fn3(x):
    if x < -999:
        y = x
    elif x > -999:
        return x
"""
  assert run_on_device_trace(code_elif_return, "jax", "torch").executed is True

  code_while_return = """
def fn4(x):
    while x > -999:
        return x
"""
  assert run_on_device_trace(code_while_return, "jax", "torch").executed is True

  code_while_false = """
def fn5(x):
    while x < -999:
        x = x + 1
    return x
"""
  assert run_on_device_trace(code_while_false, "jax", "torch").executed is True

  # Empty module (no function)
  assert run_on_device_trace("x = 1", "torch", "jax").executed is True

  # Syntax error handling
  bad_report = run_on_device_trace("def broken(:", "torch", "jax")
  assert bad_report.executed is False
  assert bad_report.error is not None


def test_autodiff_strategies_and_compiler_hooks() -> None:
  """Verify all autodiff_strategy branches, _check_all_finite, and compiler codegen integration."""
  t = OnDeviceTraceTensor((2,), data=[1.0, 2.0])
  loss_mean = _compute_loss(t, LossReduction.MEAN)
  assert loss_mean.item() == 1.5

  fake_jax = types.ModuleType("jax")
  _test_setattr(fake_jax, "value_and_grad", lambda f: f)
  fake_nnx = types.ModuleType("flax.nnx")
  _test_setattr(fake_nnx, "value_and_grad", lambda f: f)
  fake_tf = types.ModuleType("tensorflow")
  _test_setattr(fake_tf, "GradientTape", object)
  fake_mlx_nn = types.ModuleType("mlx.nn")
  _test_setattr(fake_mlx_nn, "value_and_grad", lambda m, f: f)

  mods = {
    "jax": fake_jax,
    "flax.nnx": fake_nnx,
    "tensorflow": fake_tf,
    "mlx.nn": fake_mlx_nn,
  }
  for strat in (
    "imperative_backward",
    "functional_value_and_grad",
    "nnx_value_and_grad",
    "gradient_tape",
    "mlx_value_and_grad",
    None,
  ):
    s_loss = OnDeviceTraceTensor((2,), data=[1.0, 2.0]).sum()
    assert _execute_autodiff_strategy(strat, s_loss, mods) == (strat or "compiler_grad")
    # Also test when sandbox_mods is empty
    s_loss2 = OnDeviceTraceTensor((2,), data=[1.0, 2.0]).sum()
    assert _execute_autodiff_strategy(strat, s_loss2, {}) == (strat or "compiler_grad")

  # Finite check with chex raising and with params having grad vs no grad
  p_with_grad = OnDeviceTraceTensor((2,), data=[1.0, 2.0])
  p_with_grad.grad = OnDeviceTraceTensor((2,), data=[0.5, 0.5], requires_grad=False)
  p_no_grad = OnDeviceTraceTensor((2,), data=[1.0, 2.0])
  assert _check_all_finite([t], {"w": p_with_grad, "b": p_no_grad}, {}) is True

  fake_chex = types.ModuleType("chex")

  def fail_finite(_tree: object) -> None:
    """Raise ValueError to simulate non-finite tensor detection.

    Args:
        _tree: Input tree.
    """
    raise ValueError("NaN detected")

  _test_setattr(fake_chex, "assert_tree_all_finite", fail_finite)
  assert _check_all_finite([t], {}, {"chex": fake_chex}) is False

  # Test Embedding (int32 input -> requires_grad=False so inp.grad is None)
  rep_emb = run_on_device_trace(
    """class EmbModel:
  def __init__(self):
    self.emb = nn.Embedding(16, 4)
  def forward(self, x):
    pass
    if x.sum() < -999.0:
      x = x
    if x.sum() < -999.0:
      x = x
    elif x.sum() < -999.0:
      x = x
    return relu(x.flatten())
""",
    "torch",
    "jax",
  )
  assert rep_emb.executed is True

  # Test WASM SIMD backend fallback and observed_runtime_shapes
  cfg_wasm = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    execution_backend=ExecutionBackend.WASM_SIMD,
    observed_runtime_shapes={"x": [2, 4]},
    input_shapes=[(2, 4), (2, 4)],
  )
  rep_wasm = run_on_device_trace(
    """def f(x, y):
  return x + y
""",
    "torch",
    "jax",
    config=cfg_wasm,
  )
  assert rep_wasm.forward_kernel_code is not None
  assert "WASM SIMD" in rep_wasm.forward_kernel_code
  assert "input_1" in rep_wasm.parameter_gradients

  # Test compiler integration when ml_switcheroo_compiler modules are importable
  fake_ast_to_ir = types.ModuleType("ast_to_ir")
  fake_node = mock.MagicMock()
  fake_node.id = "n0"
  fake_graph = mock.MagicMock()
  fake_graph.nodes = [fake_node]
  _test_setattr(fake_ast_to_ir, "parse_ast_to_ir", lambda _c: fake_graph)

  fake_shape_inf = types.ModuleType("shape_inference")
  _test_setattr(fake_shape_inf, "shape_inference_pass", lambda _g: None)
  _test_setattr(fake_shape_inf, "annotate_learned_shapes", lambda _g, _s: None)

  fake_autodiff = types.ModuleType("autodiff")
  _test_setattr(fake_autodiff, "generate_backward_graph", lambda g, _id: g)

  fake_webgpu = types.ModuleType("webgpu")
  mock_wgsl_cls = mock.MagicMock()
  mock_wgsl_cls.return_value.generate.return_value = "// wgsl"
  _test_setattr(fake_webgpu, "WebGPUCodeGenerator", mock_wgsl_cls)

  fake_wasm = types.ModuleType("wasm")
  mock_wasm_cls = mock.MagicMock()
  mock_wasm_cls.return_value.generate.return_value = ";; wasm"
  _test_setattr(fake_wasm, "WasmCodeGenerator", mock_wasm_cls)

  fake_dbg = types.ModuleType("shape_debugger")
  _test_setattr(fake_dbg, "format_shape_table", lambda _g: "| Compiler Table |")

  compiler_mods = {
    "ml_switcheroo_compiler.backends.ast_to_ir": fake_ast_to_ir,
    "ml_switcheroo_compiler.transforms.passes.shape_inference": fake_shape_inf,
    "ml_switcheroo_compiler.transforms.autodiff": fake_autodiff,
    "ml_switcheroo_compiler.backends.edge.webgpu": fake_webgpu,
    "ml_switcheroo_compiler.backends.edge.wasm": fake_wasm,
    "ml_switcheroo_compiler.diagnostics.shape_debugger": fake_dbg,
  }

  def fake_compiler_import(name: str) -> types.ModuleType:
    """Import stub for ml_switcheroo_compiler submodules.

    Args:
        name: Module path.

    Returns:
        Stubbed module.
    """
    if name in compiler_mods:
      return compiler_mods[name]
    raise ImportError(name)

  with mock.patch("importlib.import_module", side_effect=fake_compiler_import):
    rep_wgsl = run_on_device_trace(
      """def f(x):
  return x * 2
""",
      "torch",
      "jax",
      config=RuntimeConfig(execution_backend=ExecutionBackend.WEBGPU),
    )
    assert rep_wgsl.forward_kernel_code == "// wgsl"
    assert rep_wgsl.shape_table_markdown == "| Compiler Table |"

    fake_graph.nodes = []
    compiler_mods["ml_switcheroo_compiler.diagnostics.shape_debugger"] = types.ModuleType("empty_dbg")
    sys.modules["ml_switcheroo_compiler.debug"] = types.ModuleType("ml_switcheroo_compiler.debug")
    rep_wasm_comp = run_on_device_trace(
      """def f(x):
  return x * 2
""",
      "torch",
      "jax",
      config=RuntimeConfig(execution_backend=ExecutionBackend.WASM_SIMD),
    )
    assert rep_wasm_comp.forward_kernel_code == ";; wasm"


def test_executor_edge_branches() -> None:
  """Verify remaining edge branches in _ForwardSymbolicExecutor for 100% branch coverage."""
  t_in = OnDeviceTraceTensor((2, 4))
  ex = _ForwardSymbolicExecutor({}, [t_in], 42)
  # Test _lineno when node not in pos_map
  assert ex._lineno(cst.Name("x"), {}) == 0
  # Test _target_name fallback
  assert (
    ex._target_name(cst.Subscript(value=cst.Name("x"), slice=[cst.SubscriptElement(cst.Index(cst.Integer("0")))]))
    == "tmp"
  )
  # Test BinaryOp fallback (e.g. FloorDivide)
  val_fd, op_fd = ex._eval_expr(cst.parse_expression("x // 2"))
  assert op_fd == "BinaryOp"
  assert val_fd.shape == (2, 4)
  # Test Comparison fallback (Equal)
  val_eq, op_eq = ex._eval_expr(cst.parse_expression("x == 0"))
  assert op_eq == "Compare"
  assert val_eq.dtype == "bool"
  # Test Subscript with Slice (non-Index)
  val_sub, op_sub = ex._eval_expr(cst.parse_expression("x[0:1]"))
  assert op_sub == "Slice"
  assert val_sub.shape == (2, 4)
  # Test general Expr fallback (e.g. Lambda)
  val_lam, op_lam = ex._eval_expr(cst.parse_expression("lambda z: z"))
  assert op_lam == "Expr"
  assert val_lam.shape == (2, 4)
  # Test Call with empty leaf name
  val_call, op_call = ex._eval_call(cst.Call(func=cst.parse_expression("fn()()"), args=[]))
  assert op_call == "Call"
  assert val_call.shape == (2, 4)
  # Test method call on variable in env with unknown method name and flatten with non-int arg
  ex.env["x"] = t_in
  val_fl, _ = ex._eval_call(
    cst.Call(
      func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("flatten")),
      args=[cst.Arg(value=cst.Name("start_dim"))],
    )
  )
  assert val_fl.shape == (2, 4)
  val_unk, _ = ex._eval_call(cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("custom_method")), args=[]))
  assert val_unk.shape == (2, 4)
  # Test _parse_call_int_args with list containing -1 and non-int elements
  call_reshape = cst.parse_expression("x.reshape([-1, 2, d], other)")
  assert isinstance(call_reshape, cst.Call)
  assert ex._parse_call_int_args(call_reshape.args) == [-1, 2]
