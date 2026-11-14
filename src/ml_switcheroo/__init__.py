"""ml-switcheroo Package.

A deterministic AST transpiler for converting Deep Learning models between
frameworks (e.g., PyTorch -> JAX/Flax).

This package exposes the core compilation engine and configuration utilities
for programmatic usage.

Usage:
    Simple String Conversion::

        import ml_switcheroo as mls
        # Auto-detects defaults based on installed libraries (e.g. Torch -> JAX)
        code = "y = torch.abs(x)"
        result = mls.convert(code)
        print(result)

        # Explicit conversion
        result = mls.convert(code, source="torch", target="jax")

    Advanced Usage (AST Engine)::

        from ml_switcheroo import ASTEngine, RuntimeConfig

        config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
        engine = ASTEngine(config=config)
        res = engine.run("y = torch.abs(x)")

        if res.success:
            print(res.code)
        else:
            print(f"Errors: {res.errors}")
"""

from pydantic import BaseModel, ConfigDict

from ml_switcheroo.config import (
  ConversionMode,
  ExecutionBackend,
  OptimizationLevel,
  RuntimeConfig,
  RuntimeConfigOverrides,
)
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

__version__ = "0.0.6"


class ConvertOptions(BaseModel):
  """Optional compilation, optimization, and backend settings for ``ml_switcheroo.convert``.

  Attributes:
      plugin_settings: Optional plugin configuration flags passed to plugin hooks.
      mode: Conversion mode (``'preserve_trivia'`` or ``'optimized'``).
      opt_level: Optimization level (``'O0'``, ``'O1'``, ``'O2'``, ``'O3'``).
      backend: Execution backend identifier (defaults to ``'webgpu'``).
      on_device: Whether to execute forward and backward passes on the configured backend.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  plugin_settings: dict[str, str | int | float | bool] | None = None
  mode: str | ConversionMode = ConversionMode.PRESERVE_TRIVIA
  opt_level: str | OptimizationLevel = OptimizationLevel.O2
  backend: str | ExecutionBackend = ExecutionBackend.WEBGPU
  on_device: bool = False


def convert(
  code: str,
  source: str | None = None,
  target: str | None = None,
  strict: bool = False,
  *,
  semantics: SemanticsManager | None = None,
  options: ConvertOptions | None = None,
) -> str:
  """Transpile a string of Python code from one framework to another.

  This is a high-level convenience wrapper around the `ASTEngine`. For file-based
  conversions or batch processing, consider using `ml_switcheroo.cli` or using
  `ASTEngine` directly.

  Args:
      code: The source code to convert.
      source: The source framework key (e.g., "torch").
          If None, determined dynamically by priority via `RuntimeConfig`.
      target: The target framework key (e.g., "jax").
          If None, determined dynamically by priority via `RuntimeConfig`.
      strict: If True, the engine will return an error if an API
          cannot be mapped. If False (default), the original code
          is preserved wrapped in escape hatch comments.
      semantics: An existing Knowledge Base instance.
          If None, a new one is initialized from disk.
      options: Optional ``ConvertOptions`` bundle for plugin settings, optimization mode,
          optimization level, execution backend, and on-device tracing.

  Returns:
      The transpiled source code string.

  Raises:
      ValueError: If the conversion fails (e.g. syntax errors or strict mode violations).

  """
  opts = options if options is not None else ConvertOptions()
  config: RuntimeConfig = RuntimeConfig.load(
    RuntimeConfigOverrides(
      source=source,
      target=target,
      strict_mode=strict,
      plugin_settings=opts.plugin_settings or {},
      mode=opts.mode,
      opt_level=opts.opt_level,
      execution_backend=opts.backend,
      enable_on_device_execution=opts.on_device,
    )
  )

  manager: SemanticsManager = semantics or SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics=manager, config=config)

  result: ConversionResult = engine.run(code)

  if not result.success:
    error_msg: str = "\n".join(result.errors)
    raise ValueError(f"Transpilation failed:\n{error_msg}")

  return result.code


__all__: list[str] = [
  "ASTEngine",
  "ConversionMode",
  "ConversionResult",
  "ConvertOptions",
  "ExecutionBackend",
  "OptimizationLevel",
  "RuntimeConfig",
  "RuntimeConfigOverrides",
  "SemanticsManager",
  "__version__",
  "convert",
]
