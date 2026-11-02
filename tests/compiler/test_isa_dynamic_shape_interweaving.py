"""Unit tests for Phase 15: Interweaving Input-Dependent Shapes & Symbolic Numbers into RDNA and NVIDIA_SASS.

Tests:
- `PyTorch (Flatten + LazyLinear)` -> `RDNA` and `NVIDIA_SASS` with input `(2, 3, 4, 4)` resolving `in_features=48`.
- `Conv2d` / `Conv3d` / `AvgPool2d` / `MaxPool2d` with dynamic `(H, W)` resolving `row_stride_bytes` (`W * 4`) and `slice_stride_bytes` (`H * W * 4`).
- Non-Python sources (`RDNA`, `NVIDIA_SASS`, `StableHLO`, `MLIR`) lifted in `run_on_device_trace` and converted to `Flax NNX`, `JAX`, `PyTorch`, or cross-ISA.
- `hydrate_graph_with_runtime_trace` edge cases.
"""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.runtime_models import (
  ConstructorDimResolution,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)
from ml_switcheroo.core.runtime_tracer import (
  _lift_non_python_source_for_trace,
  hydrate_graph_with_runtime_trace,
  run_on_device_trace,
)
from ml_switcheroo.semantics.manager import SemanticsManager


def test_flatten_lazy_linear_interweaves_in_features_into_rdna_and_sass() -> None:
  """Verify Flatten + LazyLinear(10) on (2, 3, 4, 4) emits loop bound 48 in RDNA and NVIDIA_SASS."""
  py_code = """import torch.nn as nn

class DynamicProjectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten dynamic spatial dimensions
        self.flat = nn.Flatten()
        # LazyLinear depends on flattened input dimension (3 * 4 * 4 = 48)
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc(x)
        return x
"""
  sem = SemanticsManager()

  rdna_cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="rdna",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  rdna_res = ASTEngine(semantics=sem, config=rdna_cfg).run(py_code)
  assert rdna_res.success
  assert rdna_res.runtime_trace is not None
  assert rdna_res.runtime_trace.executed
  assert "s_cmp_lt_i32" in rdna_res.code and "48" in rdna_res.code
  assert "[on-device shape] in_features=48, out_features=10" in rdna_res.code
  assert "Flatten dynamic spatial dimensions" in rdna_res.code

  sass_cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="nvidia_sass",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  sass_res = ASTEngine(semantics=sem, config=sass_cfg).run(py_code)
  assert sass_res.success
  assert sass_res.runtime_trace is not None
  assert sass_res.runtime_trace.executed
  assert "isetp.lt.and" in sass_res.code.lower() and "48, pt" in sass_res.code.lower()
  assert "[on-device shape] in_features=48, out_features=10" in sass_res.code
  assert "Flatten dynamic spatial dimensions" in sass_res.code


def test_conv2d_conv3d_pool_dynamic_spatial_strides_in_sass_and_rdna() -> None:
  """Verify Conv2d, Conv3d, AvgPool2d, and MaxPool2d use runtime row_stride_bytes (W * 4) and slice_stride_bytes."""
  conv_code = """import torch.nn as nn

class SpatialConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.avgp = nn.AvgPool2d(2)
        self.maxp = nn.MaxPool2d(2)
        self.conv3 = nn.Conv3d(8, 8, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgp(x)
        x = self.maxp(x)
        x = self.conv3(x)
        return x
"""
  sem = SemanticsManager()
  sass_cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="nvidia_sass",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 6, 12)],
  )
  sass_res = ASTEngine(semantics=sem, config=sass_cfg).run(conv_code)
  assert sass_res.success
  # W=12 -> row_stride_bytes = 12 * 4 = 48; H=6, W=12 -> slice_stride_bytes = 6 * 12 * 4 = 288
  assert "48" in sass_res.code
  assert "288" in sass_res.code

  rdna_cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="rdna",
    enable_on_device_execution=True,
    input_shapes=[(2, 3, 6, 12)],
  )
  rdna_res = ASTEngine(semantics=sem, config=rdna_cfg).run(conv_code)
  assert rdna_res.success
  assert "[on-device stride] row_stride_bytes=48" in rdna_res.code


def test_non_python_sources_execute_on_device_and_interweave_shapes() -> None:
  """Verify RDNA, NVIDIA_SASS, StableHLO, and MLIR sources lift and execute on-device."""
  sem = SemanticsManager()
  rdna_src = """; RDNA Code Generation Initialized (Arch: gfx1030)
; Input x -> v0
; Custom RDNA layer comment
; BEGIN Linear (fc1)
v_mov_b32 v1, 0
s_mov_b32 s0, 0
L_GEMM_fc1:
global_load_dword v2, [v4], off
global_load_dword v3, [v5], off
s_waitcnt vmcnt(0)
v_fmac_f32 v1, v2, v3
v_add_u32 v4, v4, 4
v_add_u32 v5, v5, 4
s_add_i32 s0, s0, 1
s_cmp_lt_i32 s0, 16
s_cbranch_scc1 L_GEMM_fc1
; END Linear (fc1)
; Return: v1
"""
  cfg_rdna_to_nnx = RuntimeConfig(
    source_framework="rdna",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    input_shapes=[(2, 16)],
  )
  nnx_res = ASTEngine(semantics=sem, config=cfg_rdna_to_nnx).run(rdna_src)
  assert nnx_res.success
  assert nnx_res.runtime_trace is not None
  assert nnx_res.runtime_trace.executed
  assert "# Custom RDNA layer comment" in nnx_res.code
  assert "[on-device fwd]" in nnx_res.code

  sass_src = """// NVIDIA_SASS Code Generation
// Input x -> R0
// Custom SASS comment
// BEGIN Linear (fc1)
MOV R1, RZ;
MOV R0, RZ;
L_GEMM_fc1:
LDG.E.F32 R4, [R2];
LDG.E.F32 R5, [R3];
FFMA R1, R4, R5, R1;
IADD3 R2, R2, 4, RZ;
IADD3 R3, R3, 4, RZ;
IADD3 R0, R0, 1, RZ;
ISETP.LT.AND P0, PT, R0, 16, PT;
@P0 BRA L_GEMM_fc1;
// END Linear (fc1)
// Return: R1
"""
  cfg_sass_to_jax = RuntimeConfig(
    source_framework="nvidia_sass",
    target_framework="jax",
    enable_on_device_execution=True,
    input_shapes=[(2, 16)],
  )
  jax_res = ASTEngine(semantics=sem, config=cfg_sass_to_jax).run(sass_src)
  assert jax_res.success
  assert jax_res.runtime_trace is not None
  assert jax_res.runtime_trace.executed

  mlir_src = """module {
  sw.func @forward(%x: !sw.unknown) {
    // MLIR comment
    %0 = "sw.op"(%x) {type = "torch.relu"} : (!sw.unknown) -> !sw.unknown
    "sw.return"(%0) : (!sw.unknown) -> ()
  }
}
"""
  trace_mlir = run_on_device_trace(
    code=mlir_src,
    source_fw="mlir",
    target_fw="torch",
    semantics=sem,
    config=RuntimeConfig(
      source_framework="mlir",
      target_framework="torch",
      enable_on_device_execution=True,
      input_shapes=[(2, 8)],
    ),
  )
  assert trace_mlir.executed
  assert trace_mlir.output_shape == [2, 8]

  # Invalid non-python source fallback
  assert _lift_non_python_source_for_trace("@@@ invalid mlir @@@", "mlir", "torch", sem) is None
  bad_mlir_trace = run_on_device_trace("@@@ invalid mlir @@@", "mlir", "torch", sem)
  assert bad_mlir_trace.executed is False


def test_hydrate_graph_with_runtime_trace_branches() -> None:
  """Verify hydrate_graph_with_runtime_trace handles unexecuted traces, loops, conv ctors, and arg parsing."""
  graph = LogicalGraph(
    name="TestGraph",
    nodes={
      "x": LogicalNode(id="x", op_type="Input", attributes={}),
      "conv1": LogicalNode(id="conv1", op_type="Conv2d", attributes={}),
      "fc_positional": LogicalNode(id="fc_positional", op_type="Linear", attributes={"arg_0": "16", "arg_1": "8"}),
      "fc_bad_args": LogicalNode(id="fc_bad_args", op_type="Linear", attributes={"arg_0": "sym", "arg_1": "bad"}),
      "fc_single_arg": LogicalNode(id="fc_single_arg", op_type="Linear", attributes={"arg_0": "not_int"}),
      "output": LogicalNode(id="output", op_type="Output", attributes={}),
    },
    edges=[
      LogicalEdge("x", "conv1"),
      LogicalEdge("conv1", "fc_positional"),
      LogicalEdge("fc_positional", "fc_bad_args"),
      LogicalEdge("fc_bad_args", "fc_single_arg"),
      LogicalEdge("fc_single_arg", "output"),
    ],
  )

  # Unexecuted trace is a no-op
  unexec_trace = RuntimeTraceReport(executed=False)
  hydrate_graph_with_runtime_trace(graph, unexec_trace)
  assert graph.nodes["x"].shape_metadata is None

  exec_trace = RuntimeTraceReport(
    executed=True,
    input_shapes=[[2, 3, 8, 8]],
    output_shape=[2, 4],
    loop_trip_counts={12: 3},
    resolved_constructors={
      "conv1": ConstructorDimResolution(
        layer_name="conv1",
        op_type="Conv2d",
        inferred_in_features=3,
        out_features=6,
      ),
    },
    node_telemetry={
      "n0": NodeRuntimeTelemetry(
        node_id="n0",
        var_name="conv1",
        op_type="Conv2d",
        input_shapes=[[2, 3, 8, 8]],
        output_shape=[2, 6, 8, 8],
        grad_shape=[2, 6, 8, 8],
        grad_norm=0.5,
        is_dynamic=True,
        dynamic_reason="spatial",
      ),
    },
  )
  hydrate_graph_with_runtime_trace(graph, exec_trace)
  assert graph.nodes["conv1"].attributes["in_channels"] == 3
  assert graph.nodes["conv1"].attributes["out_channels"] == 6
  assert graph.nodes["conv1"].attributes["loop_trip_count"] == 3
  assert "dynamic: spatial" in str(graph.nodes["conv1"].attributes["runtime_comment"])
  assert graph.nodes["fc_positional"].attributes["in_features"] == 16
  assert graph.nodes["fc_positional"].attributes["out_features"] == 8
  assert isinstance(graph.nodes["fc_bad_args"].attributes["in_features"], int)

  # Test disconnected node falling back to matched_tel.input_shapes and default_in_shape, and ctor with None out_features
  graph2 = LogicalGraph(
    name="DisconnectedGraph",
    nodes={
      "orphan_with_tel": LogicalNode(id="orphan_with_tel", op_type="Linear", attributes={"arg_0": "12"}),
      "orphan_no_tel": LogicalNode(id="orphan_no_tel", op_type="Conv1d", attributes={"in_features": 4}),
      "conv_none_out": LogicalNode(id="conv_none_out", op_type="Conv2d", attributes={}),
      "lin_none_out": LogicalNode(id="lin_none_out", op_type="Linear", attributes={}),
    },
    edges=[],
  )
  trace2 = RuntimeTraceReport(
    executed=True,
    input_shapes=[[2, 3, 8]],
    output_shape=[],
    resolved_constructors={
      "conv_none_out": ConstructorDimResolution(
        layer_name="conv_none_out",
        op_type="Conv2d",
        inferred_in_features=3,
        out_features=None,
      ),
      "lin_none_out": ConstructorDimResolution(
        layer_name="lin_none_out",
        op_type="Linear",
        inferred_in_features=8,
        out_features=None,
      ),
    },
    node_telemetry={
      "orphan_with_tel": NodeRuntimeTelemetry(
        node_id="orphan_with_tel",
        var_name="orphan_with_tel",
        op_type="Linear",
        input_shapes=[[2, 8]],
        output_shape=[],
      ),
    },
  )
  hydrate_graph_with_runtime_trace(graph2, trace2)
  assert graph2.nodes["orphan_with_tel"].attributes["out_features"] == 12
  assert graph2.nodes["conv_none_out"].attributes["in_channels"] == 3
  assert graph2.nodes["lin_none_out"].attributes["in_features"] == 8
