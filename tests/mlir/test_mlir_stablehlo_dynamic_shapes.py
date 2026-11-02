"""Unit tests for Phase 16: Interweaving Ranked Tensor Types & Trivia in StableHLO and MLIR.

Tests:
- `format_mlir_tensor_type` helper across float, int, bool, and unranked shapes.
- `PyTorch/JAX -> StableHLO` with `enable_on_device_execution=True` interweaving ranked `tensor<...>` types and `// [on-device fwd]` trivia.
- `PyTorch/JAX -> MLIR` (`MlirBackend` and `PythonToMlirEmitter`) with `enable_on_device_execution=True` interweaving ranked `tensor<...>` types and `// [on-device fwd]` trivia.
- `StableHLO/MLIR -> Flax NNX / PyTorch` lifting with on-device shape interweaving and trivia preservation.
"""

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser
from ml_switcheroo.core.mlir.types import format_mlir_tensor_type
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry, RuntimeTraceReport
from ml_switcheroo.semantics.manager import SemanticsManager


def test_format_mlir_tensor_type_variants() -> None:
  """Verify format_mlir_tensor_type produces ranked and unranked tensor strings."""
  assert format_mlir_tensor_type((2, 16), "float32") == "tensor<2x16xf32>"
  assert format_mlir_tensor_type([2, 8], "int32") == "tensor<2x8xi32>"
  assert format_mlir_tensor_type((4,), "bool") == "tensor<4xi1>"
  assert format_mlir_tensor_type((1, 3, 4), "float16") == "tensor<1x3x4xf16>"
  assert format_mlir_tensor_type(None, "float32") == "tensor<*xf32>"


def test_pytorch_and_jax_to_stablehlo_interweaves_ranked_types_and_trivia() -> None:
  """Verify PyTorch and JAX -> StableHLO interweave ranked tensor<...> types and comments."""
  sem = SemanticsManager()
  py_code = """import torch
import torch.nn as nn

class RankedShloNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lazy linear layer
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        # Compute absolute value and linear projection
        y = torch.abs(x)  # inline abs comment
        z = self.fc(y)
        return z
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="stablehlo",
    enable_on_device_execution=True,
    input_shapes=[(2, 16)],
  )
  res = ASTEngine(semantics=sem, config=cfg).run(py_code)
  assert res.success
  assert res.runtime_trace is not None
  assert res.runtime_trace.executed
  assert "tensor<2x16xf32>" in res.code
  assert "tensor<2x10xf32>" in res.code
  assert "// Compute absolute value and linear projection" in res.code
  assert "// inline abs comment" in res.code
  assert "// [on-device fwd] shape=(2, 16)" in res.code
  assert "// [on-device fwd] shape=(2, 10)" in res.code


def test_pytorch_to_mlir_and_emitter_with_runtime_trace() -> None:
  """Verify PyTorch -> MLIR and PythonToMlirEmitter with runtime_trace emit ranked tensor types."""
  sem = SemanticsManager()
  py_code = """import torch.nn as nn

class MlirRankedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Init comment
        self.fc = nn.LazyLinear(6)

    def forward(self, x):
        # Forward comment
        out = self.fc(x)  # trailing comment
        return out
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="mlir",
    enable_on_device_execution=True,
    input_shapes=[(4, 12)],
  )
  res = ASTEngine(semantics=sem, config=cfg).run(py_code)
  assert res.success
  assert res.runtime_trace is not None
  assert res.runtime_trace.executed
  assert "tensor<4x12xf32>" in res.code
  assert "tensor<4x6xf32>" in res.code
  assert "// [on-device fwd] shape=(4, 6)" in res.code
  assert "// Forward comment" in res.code

  # Direct PythonToMlirEmitter and StableHloEmitter with runtime_trace
  trace = RuntimeTraceReport(
    executed=True,
    input_shapes=[[4, 12]],
    output_shape=[4, 6],
    node_telemetry={
      "n0": NodeRuntimeTelemetry(
        node_id="n0",
        var_name="out",
        op_type="Linear",
        output_shape=[4, 6],
        dtype="float32",
        grad_shape=[4, 6],
        grad_norm=1.25,
        is_dynamic=True,
        dynamic_reason="lazy",
      ),
      "n1": NodeRuntimeTelemetry(
        node_id="n1",
        var_name="other_name",
        op_type="Return",
        output_shape=[4, 6],
        dtype="float32",
      ),
    },
  )
  tree = cst.parse_module(py_code)
  emitter = PythonToMlirEmitter(runtime_trace=trace)
  mlir_text = emitter.convert(tree).to_text()
  assert "tensor<4x12xf32>" in mlir_text
  assert "tensor<4x6xf32>" in mlir_text
  assert "dynamic: lazy" in mlir_text
  assert "[bwd] grad=(4, 6)" in mlir_text

  # Exhaust cursor fallback branch and _format_mlir_runtime_comment without grad_shape
  assert emitter._match_stmt_telemetry("nonexistent") is None
  assert "[bwd]" not in PythonToMlirEmitter._format_mlir_runtime_comment(trace.node_telemetry["n1"])

  shlo_emitter = StableHloEmitter(semantics=sem, runtime_trace=trace)
  shlo_text = shlo_emitter.convert(tree).to_text()
  assert "tensor<4x12xf32>" in shlo_text
  assert "tensor<4x6xf32>" in shlo_text
  assert shlo_emitter._extract_literal(cst.parse_expression("-5")) == -5

  # Also test StableHloEmitter without runtime_trace where infer_pass.return_types is used
  static_fn = cst.parse_module("""def fn(x: tensor):
    return x
""")
  static_shlo = StableHloEmitter(semantics=sem).convert(static_fn).to_text()
  assert "func.func" in static_shlo

  # Also test expression statement, tuple target, and unmatched variable in inference function with runtime_trace
  extra_tree = cst.parse_module("""def forward(x):
    1 + 1
    (a, b) = (1, 2)
    unmatched = x + 1
    return x
""")
  extra_emitter = PythonToMlirEmitter(runtime_trace=trace)
  assert extra_emitter.convert(extra_tree) is not None


def test_stablehlo_and_mlir_lifting_to_frameworks_with_on_device_execution() -> None:
  """Verify StableHLO and MLIR lift to Python frameworks with comments and on-device telemetry."""
  sem = SemanticsManager()
  shlo_code = """module {
  func.func @forward(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
    // StableHLO user comment before abs
    %0 = stablehlo.abs %x : tensor<2x8xf32>
    // StableHLO custom call comment
    %1 = stablehlo.custom_call @relu(%0) {call_target_name = "@relu"} : (tensor<2x8xf32>) -> tensor<2x8xf32>
    func.return %1 : tensor<2x8xf32>
  }
}
"""
  parsed = StableHloParser(shlo_code).parse()
  py_mod = MlirToPythonGenerator().generate(parsed)
  assert "# StableHLO user comment before abs" in py_mod.code
  assert "# StableHLO custom call comment" in py_mod.code

  cfg_shlo_to_torch = RuntimeConfig(
    source_framework="stablehlo",
    target_framework="torch",
    enable_on_device_execution=True,
    input_shapes=[(2, 8)],
  )
  torch_res = ASTEngine(semantics=sem, config=cfg_shlo_to_torch).run(shlo_code)
  assert torch_res.success
  assert torch_res.runtime_trace is not None
  assert torch_res.runtime_trace.executed
  assert "# StableHLO user comment before abs" in torch_res.code
  assert "[on-device fwd] shape=(2, 8)" in torch_res.code
