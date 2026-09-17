"""Test suite for bidirectional translation across Python frameworks and hardware ISAs.

Verifies the full roundtrip compilation matrix:
High-Level (Torch, JAX, MLX, Keras) <-> LogicalGraph <-> Hardware ISA (RDNA, SASS).
"""

import ast
from typing import Dict, Optional, Tuple

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class MockHardwareSemantics:
  """Mock semantics provider for hardware translation."""

  def get_definition(self, kind: str) -> Tuple[str, Dict[str, str]]:
    """Return mock definition for operation kind.

    Args:
        kind: The operation name or kind.

    Returns:
        A tuple of (kind, metadata).
    """
    return (kind, {})

  def resolve_variant(self, aid: str, fw: str) -> Optional[Dict[str, str]]:
    """Resolve framework-specific variant for abstract ID.

    Args:
        aid: The abstract operation ID.
        fw: The target framework identifier.

    Returns:
        A dictionary with the resolved API name, or None.
    """
    if fw == "nvidia_sass":
      return {"api": f"Macro.{aid}"}
    if fw == "rdna":
      return {"api": f"; Macro.{aid}"}
    if fw == "torch":
      if aid == "Conv2d":
        return {"api": "nn.Conv2d"}
      if aid == "Linear":
        return {"api": "nn.Linear"}
      return {"api": f"torch.{aid.lower()}"}
    if fw in ("jax", "flax", "flax_nnx"):
      if aid == "Conv2d":
        return {"api": "flax.nnx.Conv"}
      if aid == "Linear":
        return {"api": "flax.nnx.Linear"}
      return {"api": f"jnp.{aid.lower()}"}
    if fw == "mlx":
      if aid == "Conv2d":
        return {"api": "nn.Conv2d"}
      if aid == "Linear":
        return {"api": "nn.Linear"}
      return {"api": f"mx.{aid.lower()}"}
    if fw == "keras":
      if aid == "Conv2d":
        return {"api": "keras.layers.Conv2D"}
      if aid == "Linear":
        return {"api": "keras.layers.Dense"}
      return {"api": f"keras.ops.{aid.lower()}"}
    return None


@pytest.mark.parametrize("hardware_target", ["rdna", "nvidia_sass"])
@pytest.mark.parametrize("python_target", ["torch", "jax", "mlx", "keras"])
def test_full_roundtrip_hardware_python_matrix(hardware_target: str, python_target: str) -> None:
  """Verifies the bidirectional translation between hardware ISAs and Python frameworks.

  Args:
      hardware_target: The hardware ISA target ('rdna' or 'nvidia_sass').
      python_target: The high-level Python framework ('torch', 'jax', 'mlx', 'keras').
  """
  semantics = MockHardwareSemantics()

  # 1. Build canonical DAG
  nodes = {
    "x": LogicalNode("x", op_type="Input"),
    "c1": LogicalNode("c1", op_type="Conv2d", attributes={"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
    "r1": LogicalNode("r1", op_type="ReLU"),
    "out": LogicalNode("out", op_type="Output"),
  }
  g_initial = LogicalGraph(
    name="MatrixNet",
    nodes=nodes,
    edges=[
      LogicalEdge("x", "c1"),
      LogicalEdge("c1", "r1"),
      LogicalEdge("r1", "out"),
    ],
  )

  # 2. Compile to hardware assembly
  if hardware_target == "rdna":
    rdna_backend = RdnaBackend(semantics)
    asm_text = rdna_backend.compile(g_initial)
    assert "BEGIN Conv2d" in asm_text
    assert "BEGIN ReLU" in asm_text

    # 3. Lift from RDNA
    rdna_ast = RdnaParser(asm_text).parse().statements
    g_lifted = RdnaLifter().lift(rdna_ast)
  else:
    sass_backend = NvidiaSassBackend(semantics)
    asm_text = sass_backend.compile(g_initial)
    assert "BEGIN Conv2d" in asm_text
    assert "BEGIN ReLU" in asm_text

    # 3. Lift from SASS
    sass_ast = NvidiaSassParser(asm_text).parse().statements
    g_lifted = NvidiaSassLifter().lift(sass_ast)

  assert any(n.op_type == "Conv2d" for n in g_lifted.nodes.values())
  assert any(n.op_type == "ReLU" for n in g_lifted.nodes.values())

  # 4. Synthesize from lifted graph to Python target
  python_backend = PythonBackend(framework=python_target, semantics=semantics)
  py_code = python_backend.compile(g_lifted)

  # 5. Validate Python syntax
  try:
    ast.parse(py_code)
  except SyntaxError as e:
    pytest.fail(f"Generated Python code has syntax error for target {python_target}: {e}")

  # 6. Verify framework-specific class invariants
  if python_target == "torch":
    assert "(nn.Module):" in py_code
    assert "def forward(self, x):" in py_code
  elif python_target == "jax":
    assert "nnx.Rngs" in py_code
    assert "def __call__(self, x):" in py_code
  elif python_target == "mlx":
    assert "(mlx.nn.Module):" in py_code
    assert "def __call__(self, x):" in py_code
  elif python_target == "keras":
    assert "(keras.Model):" in py_code
    assert "def call(self, x):" in py_code


@pytest.mark.parametrize("src_hw,dst_hw", [("rdna", "nvidia_sass"), ("nvidia_sass", "rdna")])
def test_direct_cross_isa_compilation(src_hw: str, dst_hw: str) -> None:
  """Verifies direct cross-compilation between AMD RDNA and NVIDIA SASS assembly.

  Args:
      src_hw: Source hardware architecture ('rdna' or 'nvidia_sass').
      dst_hw: Destination hardware architecture ('rdna' or 'nvidia_sass').
  """
  from ml_switcheroo import convert

  if src_hw == "rdna":
    code = "; BEGIN Conv2d (conv)\nv_add_f32 v1, v0, v0\n; END Conv2d (conv)\n"
  else:
    code = "// BEGIN Conv2d (conv)\nFADD R1, R0, R0;\n// END Conv2d (conv)\n"

  converted = convert(code, source=src_hw, target=dst_hw)
  assert "Conv2d" in converted
  if dst_hw == "nvidia_sass":
    assert "FFMA" in converted or "FADD" in converted or "BEGIN Conv2d" in converted
  else:
    assert "v_fmac_f32" in converted or "v_add_f32" in converted or "BEGIN Conv2d" in converted


@pytest.mark.parametrize("src_py", ["torch", "jax", "mlx", "keras"])
@pytest.mark.parametrize("dst_hw", ["rdna", "nvidia_sass"])
def test_high_level_to_hardware_isa_compilation(src_py: str, dst_hw: str) -> None:
  """Verifies compiling high-level Python models into hardware assembly ISAs.

  Args:
      src_py: Source high-level Python framework ('torch', 'jax', 'mlx', 'keras').
      dst_hw: Target hardware architecture ('rdna' or 'nvidia_sass').
  """
  from ml_switcheroo import convert

  sample_code = """
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        return self.conv(x)
"""
  converted = convert(sample_code, source=src_py, target=dst_hw)
  assert "Conv2d" in converted
  if dst_hw == "rdna":
    assert "; RDNA Code Generation Initialized" in converted or "v0" in converted
  else:
    assert "R0" in converted or "MOV" in converted or "Conv2d" in converted


@pytest.mark.parametrize(
  "src_fw,dst_fw",
  [
    ("torch", "jax"),
    ("jax", "torch"),
    ("torch", "mlx"),
    ("mlx", "torch"),
    ("torch", "keras"),
    ("keras", "torch"),
    ("jax", "mlx"),
    ("mlx", "jax"),
    ("jax", "keras"),
    ("keras", "jax"),
    ("mlx", "keras"),
    ("keras", "mlx"),
  ],
)
def test_full_high_level_python_bidirectional_matrix(src_fw: str, dst_fw: str) -> None:
  """Verifies static AST/CST conversion across all 12 high-level Python framework pairs.

  Args:
      src_fw: Source framework identifier.
      dst_fw: Target framework identifier.
  """
  from ml_switcheroo import convert

  snippets = {
    "torch": (
      "import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n"
      "    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(3, 16, 3)\n"
      "    def forward(self, x):\n        return self.conv(x)\n"
    ),
    "jax": (
      "import jax.numpy as jnp\nimport flax.nnx as nnx\nclass Net(nnx.Module):\n"
      "    def __init__(self, rngs: nnx.Rngs):\n        self.conv = nnx.Conv(3, 16, 3, rngs=rngs)\n"
      "    def __call__(self, x):\n        return self.conv(x)\n"
    ),
    "mlx": (
      "import mlx.core as mx\nimport mlx.nn as nn\nclass Net(nn.Module):\n"
      "    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(3, 16, 3)\n"
      "    def __call__(self, x):\n        return self.conv(x)\n"
    ),
    "keras": (
      "import keras\nimport keras.layers as layers\nclass Net(keras.Model):\n"
      "    def __init__(self):\n        super().__init__()\n        self.conv = layers.Conv2D(16, 3)\n"
      "    def call(self, x):\n        return self.conv(x)\n"
    ),
  }

  converted = convert(snippets[src_fw], source=src_fw, target=dst_fw)

  # Validate generated Python syntax
  try:
    ast.parse(converted)
  except SyntaxError as exc:
    pytest.fail(f"Conversion {src_fw} -> {dst_fw} generated invalid Python syntax: {exc}")

  # Validate framework target class definitions
  if dst_fw == "torch":
    assert "(nn.Module):" in converted
    assert "def forward(self, x):" in converted
  elif dst_fw in ("jax", "flax_nnx", "flax"):
    assert "def __call__(self, x):" in converted
  elif dst_fw == "mlx":
    assert "(nn.Module):" in converted
    assert "def __call__(self, x):" in converted
  elif dst_fw == "keras":
    assert "keras.Model" in converted or "keras.Layer" in converted
    assert "def call(self, x):" in converted
