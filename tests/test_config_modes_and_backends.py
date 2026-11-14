"""Unit tests for ConversionMode, OptimizationLevel, and ExecutionBackend across Config, SDK, and CLI."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import ml_switcheroo as mls
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.config import (
  ConversionMode,
  ExecutionBackend,
  OptimizationLevel,
  RuntimeConfig,
  RuntimeConfigOverrides,
)
from ml_switcheroo.core.engine import ASTEngine


def test_conversion_mode_and_optimization_level_defaults() -> None:
  """Verify RuntimeConfig defaults to preserve_trivia, O2, and webgpu."""
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert cfg.mode == ConversionMode.PRESERVE_TRIVIA
  assert cfg.opt_level == OptimizationLevel.O2
  assert cfg.execution_backend == ExecutionBackend.WEBGPU
  assert cfg.enable_graph_optimization is False
  assert cfg.emit_backward_code is False


def test_mode_and_graph_optimization_synchronization() -> None:
  """Verify two-way synchronization between ConversionMode and enable_graph_optimization."""
  cfg_opt = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    mode=ConversionMode.OPTIMIZED,
  )
  assert cfg_opt.enable_graph_optimization is True

  cfg_legacy = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    enable_graph_optimization=True,
  )
  assert cfg_legacy.mode == ConversionMode.OPTIMIZED


def test_execution_backend_catalog_values() -> None:
  """Verify all ml-switcheroo-compiler backends are exposed on ExecutionBackend."""
  expected = {
    "webgpu",
    "wasm_simd",
    "webgl",
    "pyodide",
    "auto",
    "onnx",
    "stablehlo",
    "cuda",
    "rocm",
    "metal",
    "llvm_cpp",
    "jax",
    "pytorch",
    "torch",
    "mlx",
    "tensorflow",
    "keras",
    "numpy",
    "cupy",
    "dask",
    "numba",
    "dpnp",
    "sparse",
    "sparse_coo",
    "awkward",
    "pyarrow_compute",
    "pure_python",
  }
  actual = {b.value for b in ExecutionBackend}
  assert expected.issubset(actual)
  assert ExecutionBackend("webgpu") == ExecutionBackend.WEBGPU


def test_runtime_config_load_modes_and_backends(tmp_path: Path) -> None:
  """Verify RuntimeConfig.load handles CLI overrides and pyproject.toml settings for mode, opt_level, and backend.

  Args:
      tmp_path: Temporary directory pytest fixture.
  """
  toml_path = tmp_path / "pyproject.toml"
  toml_path.write_text(
    """[tool.ml_switcheroo]
source_framework = "torch"
target_framework = "jax"
mode = "optimized"
opt_level = "O3"
execution_backend = "cuda"
enable_on_device_execution = true
emit_backward_code = true
""",
    encoding="utf-8",
  )

  # Load from TOML
  cfg_toml = RuntimeConfig.load(search_path=tmp_path)
  assert cfg_toml.mode == ConversionMode.OPTIMIZED
  assert cfg_toml.enable_graph_optimization is True
  assert cfg_toml.opt_level == OptimizationLevel.O3
  assert cfg_toml.execution_backend == ExecutionBackend.CUDA
  assert cfg_toml.enable_on_device_execution is True
  assert cfg_toml.emit_backward_code is True

  # Override TOML via explicit arguments
  cfg_override = RuntimeConfig.load(
    RuntimeConfigOverrides(
      search_path=tmp_path,
      mode=ConversionMode.PRESERVE_TRIVIA,
      opt_level=OptimizationLevel.O1,
      execution_backend=ExecutionBackend.WASM_SIMD,
      enable_on_device_execution=False,
      emit_backward_code=False,
    )
  )
  assert cfg_override.mode == ConversionMode.PRESERVE_TRIVIA
  assert cfg_override.enable_graph_optimization is False
  assert cfg_override.opt_level == OptimizationLevel.O1
  assert cfg_override.execution_backend == ExecutionBackend.WASM_SIMD
  assert cfg_override.enable_on_device_execution is False
  assert cfg_override.emit_backward_code is False

  # Explicit mode + explicit enable_graph_optimization
  cfg_both = RuntimeConfig.load(
    RuntimeConfigOverrides(
      search_path=tmp_path,
      mode="preserve_trivia",
      enable_graph_optimization=False,
    )
  )
  assert cfg_both.mode == ConversionMode.PRESERVE_TRIVIA
  assert cfg_both.enable_graph_optimization is False


def test_ast_engine_and_sdk_convert_with_mode_and_backend() -> None:
  """Verify ASTEngine and ml_switcheroo.convert accept mode, opt_level, and backend."""
  base_cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    mode=ConversionMode.OPTIMIZED,
    opt_level=OptimizationLevel.O3,
    execution_backend=ExecutionBackend.METAL,
  )
  eng = ASTEngine(config=base_cfg)
  assert eng.config.mode == ConversionMode.OPTIMIZED
  assert eng.config.enable_graph_optimization is True
  assert eng.config.opt_level == OptimizationLevel.O3
  assert eng.config.execution_backend == ExecutionBackend.METAL

  eng2 = ASTEngine.from_options(
    RuntimeConfigOverrides(
      source="torch",
      target="jax",
      mode="preserve_trivia",
      opt_level="O1",
      execution_backend="webgpu",
    )
  )
  assert eng2.config.mode == ConversionMode.PRESERVE_TRIVIA
  assert eng2.config.opt_level == OptimizationLevel.O1
  assert eng2.config.execution_backend == ExecutionBackend.WEBGPU

  code = """# Header trivia comment

import torch

def fn(x):
    # Inline trivia comment
    return torch.abs(x)
"""
  out = mls.convert(
    code,
    source="torch",
    target="jax",
    options=mls.ConvertOptions(
      mode="preserve_trivia",
      opt_level="O2",
      backend="webgpu",
    ),
  )
  assert "# Header trivia comment" in out
  assert "# Inline trivia comment" in out


def test_cli_convert_mode_optimize_and_backend_flags(tmp_path: Path) -> None:
  """Verify CLI convert flags --mode, --optimize, -O/--opt-level, --backend, and --on-device.

  Args:
      tmp_path: Temporary directory pytest fixture.
  """
  src_file = tmp_path / "model.py"
  src_file.write_text(
    """import torch
def f(x):
    return torch.abs(x)
""",
    encoding="utf-8",
  )
  out_file = tmp_path / "out.py"

  with mock.patch("ml_switcheroo.cli.commands.handle_convert", return_value=0) as mock_conv:
    rc = main(
      [
        "convert",
        str(src_file),
        "--out",
        str(out_file),
        "--source",
        "torch",
        "--target",
        "jax",
        "--optimize",
        "-O",
        "O3",
        "--backend",
        "webgpu",
        "--on-device",
      ]
    )
    assert rc == 0
    opts = mock_conv.call_args[0][0]
    assert opts.mode == "optimized"
    assert opts.opt_level == "O3"
    assert opts.execution_backend == "webgpu"
    assert opts.enable_on_device_execution is True
