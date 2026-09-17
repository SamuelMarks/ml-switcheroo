"""Integration tests for ASTEngine with Intermediate Representation (IR)."""

import json
from typing import Any, Dict
from unittest.mock import patch
import pytest


from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.engine import ASTEngine

SAMPLE_TORCH_CODE = """
import torch.nn as nn

class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)
"""

SAMPLE_JAX_CODE = """
import jax.numpy as jnp
from flax import nnx

class SampleJaxNet(nnx.Module):
    def __init__(self, rngs):
        self.conv = nnx.Conv(3, 16, (3, 3), rngs=rngs)

    def __call__(self, x):
        return self.conv(x)
"""

SAMPLE_IR_JSON = """{
  "name": "LinearNet",
  "nodes": [
    {"id": "x", "kind": "Input"},
    {"id": "fc1", "kind": "Linear", "inputs": ["x"]}
  ],
  "edges": [{"source": "x", "target": "fc1"}]
}"""


def test_engine_torch_to_ir() -> None:
  """Test converting PyTorch model code to IR JSON."""
  engine = ASTEngine(source="torch", target="ir")
  res = engine.run(SAMPLE_TORCH_CODE)
  assert res.success
  data: Dict[str, Any] = json.loads(res.code)
  assert data["name"] == "SampleModel"
  assert len(data["nodes"]) >= 1
  assert any(n["id"] == "conv" for n in data["nodes"])


def test_engine_jax_to_ir() -> None:
  """Test converting JAX model code to IR JSON."""
  engine = ASTEngine(source="jax", target="ir")
  res = engine.run(SAMPLE_JAX_CODE)
  assert res.success
  data: Dict[str, Any] = json.loads(res.code)
  assert data["name"] == "SampleJaxNet"
  assert len(data["nodes"]) >= 1


def test_engine_ir_to_torch() -> None:
  """Test converting IR JSON to PyTorch code."""
  engine = ASTEngine(source="ir", target="torch")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert "class LinearNet(nn.Module):" in res.code
  assert "self.fc1 = nn.Linear()" in res.code
  assert "self.fc1(x)" in res.code
  assert "return x" in res.code


def test_engine_ir_to_jax() -> None:
  """Test converting IR JSON to JAX code."""
  engine = ASTEngine(source="ir", target="jax")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_ir_to_keras() -> None:
  """Test converting IR JSON to Keras code."""
  engine = ASTEngine(source="ir", target="keras")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_ir_to_mlx() -> None:
  """Test converting IR JSON to MLX code."""
  engine = ASTEngine(source="ir", target="mlx")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_ir_to_stablehlo() -> None:
  """Test compiling IR JSON directly to StableHLO MLIR text."""
  engine = ASTEngine(source="ir", target="stablehlo")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_ir_to_nvidia_sass() -> None:
  """Test compiling IR JSON directly to NVIDIA SASS assembly."""
  engine = ASTEngine(source="ir", target="nvidia_sass")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_ir_to_rdna() -> None:
  """Test compiling IR JSON directly to AMD RDNA assembly."""
  engine = ASTEngine(source="ir", target="rdna")
  res = engine.run(SAMPLE_IR_JSON)
  assert res.success
  assert len(res.code) > 0


def test_engine_intermediate_ir_mode() -> None:
  """Test two-hop conversion using --intermediate ir."""
  engine = ASTEngine(source="torch", target="jax", intermediate="ir")
  res = engine.run(SAMPLE_TORCH_CODE)
  assert res.success
  assert len(res.code) > 0


def test_engine_intermediate_ir_failure_propagation() -> None:
  """Test error propagation when intermediate conversion hop fails."""
  engine = ASTEngine(source="torch", target="jax", intermediate="ir")
  with patch.object(
    ASTEngine,
    "_run_compiler_pipeline",
    return_value=ConversionResult(code="", success=False, errors=["Simulated failure"]),
  ):
    res = engine.run(SAMPLE_TORCH_CODE)
    assert not res.success
    assert "Simulated failure" in res.errors[0]


def test_ingest_code_ir_source() -> None:
  """Test ingest_code with IR source constructing LibCST tree."""
  from ml_switcheroo.core.ingestion import ingest_code
  from ml_switcheroo.core.tracer import get_tracer
  from ml_switcheroo.frameworks.ir import IrAdapter

  adapter = IrAdapter()
  tracer = get_tracer()
  tree = ingest_code(SAMPLE_IR_JSON, "ir", "torch", adapter, tracer)
  assert tree is not None

  # Test failure branch
  with pytest.raises(Exception):
    ingest_code("{bad_json:", "ir", "torch", adapter, tracer)


def test_engine_stablehlo_to_rdna() -> None:
  """Test converting StableHLO to RDNA in ASTEngine compiler pipeline."""
  stablehlo_code = """
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
"""
  engine = ASTEngine(source="stablehlo", target="rdna")
  res = engine.run(stablehlo_code)
  assert res.success
  assert len(res.code) > 0
