"""Unit tests for `src/ml_switcheroo/core/compiler/backends/compiler_bridge.py`."""

from __future__ import annotations

from unittest import mock

from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

from ml_switcheroo.core.compiler.backends.compiler_bridge import (
  BRIDGED_COMPILER_TARGETS,
  CompilerGraphBackendAdapter,
  compile_logical_graph_with_backend,
  resolve_compiler_registry_key,
)
from ml_switcheroo.core.compiler.registry import get_backend_class, instantiate_backend, is_isa_target


def _sample_graph() -> LogicalGraph:
  """Create a minimal `LogicalGraph` for backend compilation tests.

  Returns:
      Populated `LogicalGraph`.
  """
  g = LogicalGraph(name="BridgeNet")
  g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["y"] = LogicalNode(id="y", op_type="Relu", shape_metadata=(2, 4), dtype=DType.float32)
  g.edges.append(LogicalEdge(source="x", target="y"))
  g.inputs = ["x"]
  g.outputs = ["y"]
  return g


def test_resolve_compiler_registry_key() -> None:
  """Verify alias resolution for webgpu, wasm_simd, webgl, onnx, and direct targets."""
  assert resolve_compiler_registry_key("webgpu") == "webgpu"
  assert resolve_compiler_registry_key("AUTO") == "webgpu"
  assert resolve_compiler_registry_key("pyodide") == "webgpu"
  assert resolve_compiler_registry_key("wasm_simd") == "edge_wasm_simd"
  assert resolve_compiler_registry_key("wasm") == "edge_wasm_simd"
  assert resolve_compiler_registry_key("webgl") == "edge_webgl"
  assert resolve_compiler_registry_key("onnx") == "edge_onnx"
  assert resolve_compiler_registry_key("cuda") == "cuda"


def test_compile_logical_graph_with_backend_live_and_fallback() -> None:
  """Verify compile_logical_graph_with_backend across live BackendRegistry and fallbacks."""
  g = _sample_graph()
  for target in ("webgpu", "wasm_simd", "webgl", "cuda", "metal", "rocm", "llvm_cpp", "pure_python"):
    code = compile_logical_graph_with_backend(g, target=target)
    assert isinstance(code, str)
    assert len(code) > 0

  # Test fallback when BackendRegistry import fails
  with mock.patch("importlib.import_module", side_effect=ImportError("no compiler")):
    wasm_fallback = compile_logical_graph_with_backend(g, target="wasm_simd")
    assert "(module" in wasm_fallback
    assert "forward_pass" in wasm_fallback

    wgsl_fallback = compile_logical_graph_with_backend(g, target="webgpu")
    assert "@compute @workgroup_size(64)" in wgsl_fallback


def test_compiler_graph_backend_adapter_and_registry() -> None:
  """Verify CompilerGraphBackendAdapter registration and instantiation via compiler registry."""
  g = _sample_graph()
  for tgt in BRIDGED_COMPILER_TARGETS:
    cls = get_backend_class(tgt)
    assert cls is CompilerGraphBackendAdapter
    assert is_isa_target(tgt) is True
    inst = instantiate_backend(cls, target=tgt)
    assert isinstance(inst, CompilerGraphBackendAdapter)
    assert inst.target == tgt

  adapter = CompilerGraphBackendAdapter(target="webgpu")
  compiled = adapter.compile(g)
  assert isinstance(compiled, str)
  assert len(compiled) > 0
