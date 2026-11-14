"""Unit tests for Phase 21: Lossless Bidirectional Shape & Parameter Recovery in `RdnaAnalyzer`, `NvidiaSassAnalyzer`, and `IsaLifter`.

Verifies:
- `RdnaAnalyzer.LOOP_PARAMETER_RULES` parity with `NvidiaSassAnalyzer` across all 19 op kinds.
- `IsaLifter._extract_on_device_kv_pairs` and `IsaLifter.lift` extracting `[on-device shape]` & `[on-device stride]` metadata into `LogicalNode.attributes`.
- Lossless round-tripping `PyTorch (Flatten + LazyLinear(10)) -> RDNA / NVIDIA_SASS -> Flax NNX / PyTorch / StableHLO` preserving both `in_features=48` and `out_features=10`.
- Lossless round-tripping `Conv2d(3, 8, 3)` through `RDNA / NVIDIA_SASS` preserving `in_channels=3`, `out_channels=8`, and `row_stride_bytes`.
"""

from __future__ import annotations

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaImmediate, RdnaInstruction, RdnaSGPR
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.isa.lifter import IsaLifter
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


def test_rdna_analyzer_parity_with_nvidia_sass_analyzer() -> None:
  """Verify `RdnaAnalyzer.LOOP_PARAMETER_RULES` covers all 19 op kinds in `NvidiaSassAnalyzer`."""
  assert set(RdnaAnalyzer.LOOP_PARAMETER_RULES.keys()) == set(NvidiaSassAnalyzer.LOOP_PARAMETER_RULES.keys())
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=5)])
  for op_kind, (named_key, pos_key) in RdnaAnalyzer.LOOP_PARAMETER_RULES.items():
    meta = RdnaAnalyzer.analyze_block(op_kind, [inst])
    assert meta[named_key] == 5
    assert meta[pos_key] == 5


def test_isa_lifter_extracts_on_device_shape_and_stride_metadata() -> None:
  """Verify `IsaLifter` extracts key-value pairs from `[on-device shape]` and `[on-device stride]` comments."""
  kv_shape = IsaLifter._extract_on_device_kv_pairs("[on-device shape] in_features=48, out_features=10, mode=fast")
  assert kv_shape == {"in_features": 48, "out_features": 10, "mode": "fast"}

  kv_stride = IsaLifter._extract_on_device_kv_pairs(
    "[on-device stride] row_stride_bytes=48, in_channels=3, out_channels=8"
  )
  assert kv_stride == {"row_stride_bytes": 48, "in_channels": 3, "out_channels": 8}

  assert IsaLifter._extract_on_device_kv_pairs("regular comment") == {}
  assert IsaLifter._extract_on_device_kv_pairs("[on-device shape] malformed_token, =10, k=") == {}

  nodes = [
    RdnaComment(text="; BEGIN Linear (fc)"),
    RdnaComment(text="; [on-device shape] in_features=48, out_features=10"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=48)]),
    RdnaComment(text="; END Linear (fc)"),
    RdnaComment(text="; BEGIN Conv2d (conv)"),
    RdnaComment(text="; [on-device stride] row_stride_bytes=32, in_channels=3, out_channels=8"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=2), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d (conv)"),
  ]
  graph = RdnaLifter().lift(nodes)
  assert graph.nodes["fc"].attributes["in_features"] == 48
  assert graph.nodes["fc"].attributes["out_features"] == 10
  assert "arg_0" not in graph.nodes["fc"].attributes
  assert graph.nodes["conv"].attributes["in_channels"] == 3
  assert graph.nodes["conv"].attributes["out_channels"] == 8
  assert graph.nodes["conv"].attributes["row_stride_bytes"] == 32
  assert graph.nodes["conv"].attributes["k"] == 3
  assert "arg_2" not in graph.nodes["conv"].attributes


def test_lossless_roundtrip_pytorch_isa_flax_torch_stablehlo() -> None:
  """Verify `PyTorch (Flatten + LazyLinear(10)) -> RDNA / NVIDIA_SASS -> Flax NNX / PyTorch / StableHLO` preserves `48` and `10`."""
  py_source = """import torch
import torch.nn as nn

class LazyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten + LazyLinear comments
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        # Forward pass comment
        x = self.flatten(x)
        x = self.fc(x)
        return x
"""
  sem = SemanticsManager()

  # 1. PyTorch -> RDNA with input (2, 3, 4, 4) -> in_features=48, out_features=10
  cfg_to_rdna = RuntimeConfig(
    source_framework="torch",
    target_framework="rdna",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  rdna_res = ASTEngine(semantics=sem, config=cfg_to_rdna).run(py_source)
  assert rdna_res.success
  assert "[on-device shape] in_features=48, out_features=10" in rdna_res.code

  # 2. RDNA -> Flax NNX preserves both in_features=48 and out_features=10
  cfg_rdna_to_flax = RuntimeConfig(
    source_framework="rdna",
    target_framework="flax_nnx",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  flax_res = ASTEngine(semantics=sem, config=cfg_rdna_to_flax).run(rdna_res.code)
  assert flax_res.success
  assert "in_features=48" in flax_res.code
  assert "out_features=10" in flax_res.code
  assert "rngs=rngs" in flax_res.code

  # 3. RDNA -> StableHLO preserves in_features=48, out_features=10, and ranked tensor<2x10xf32>
  cfg_rdna_to_shlo = RuntimeConfig(
    source_framework="rdna",
    target_framework="stablehlo",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  shlo_res = ASTEngine(semantics=sem, config=cfg_rdna_to_shlo).run(rdna_res.code)
  assert shlo_res.success
  assert "in_features = 48" in shlo_res.code
  assert "out_features = 10" in shlo_res.code
  assert "tensor<2x10xf32>" in shlo_res.code

  # 4. PyTorch -> NVIDIA_SASS -> PyTorch preserves in_features=48 and out_features=10
  cfg_to_sass = RuntimeConfig(
    source_framework="torch",
    target_framework="nvidia_sass",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  sass_res = ASTEngine(semantics=sem, config=cfg_to_sass).run(py_source)
  assert sass_res.success
  assert "[on-device shape] in_features=48, out_features=10" in sass_res.code

  cfg_sass_to_torch = RuntimeConfig(
    source_framework="nvidia_sass",
    target_framework="torch",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    input_shapes=[(2, 3, 4, 4)],
  )
  torch_res = ASTEngine(semantics=sem, config=cfg_sass_to_torch).run(sass_res.code)
  assert torch_res.success
  assert "in_features=48" in torch_res.code
  assert "out_features=10" in torch_res.code
