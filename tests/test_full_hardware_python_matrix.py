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
  g_initial = LogicalGraph(
    name="MatrixNet",
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("c1", "Conv2d", {"in_channels": 3, "out_channels": 16, "kernel_size": 3}),
      LogicalNode("r1", "ReLU"),
      LogicalNode("out", "Output"),
    ],
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

  assert any(n.kind == "Conv2d" for n in g_lifted.nodes)
  assert any(n.kind == "ReLU" for n in g_lifted.nodes)

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
