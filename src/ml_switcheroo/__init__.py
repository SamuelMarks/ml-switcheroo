"""
ml-switcheroo Package.

A deterministic AST transpiler for converting Deep Learning models between
frameworks (e.g., PyTorch -> JAX/Flax).

This package exposes the core compilation engine and configuration utilities
for programmatic usage.

Usage
-----

Simple String Conversion
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import ml_switcheroo as mls
    code = "y = torch.abs(x)"
    result = mls.convert(code, source="torch", target="jax")
    print(result)
    # y = jax.numpy.abs(x)

Advanced Usage (AST Engine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from ml_switcheroo import ASTEngine, RuntimeConfig

    config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
    engine = ASTEngine(config=config)
    res = engine.run("y = torch.abs(x)")

    if res.success:
        print(res.code)
    else:
        print(f"Errors: {res.errors}")
"""

from typing import Any, Dict, Optional

# Core Engine Components
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

__version__ = "0.0.1"


def convert(
  code: str,
  source: str = "torch",
  target: str = "jax",
  strict: bool = False,
  plugin_settings: Optional[Dict[str, Any]] = None,
  semantics: Optional[SemanticsManager] = None,
) -> str:
  """
  Transpiles a string of Python code from one framework to another.

  This is a high-level convenience wrapper around the `ASTEngine`. For file-based
  conversions or batch processing, consider using `ml_switcheroo.cli` or using
  `ASTEngine` directly.

  Args:
      code (str): The source code to convert.
      source (str): The source framework key (e.g., "torch", "jax").
      target (str): The target framework key (e.g., "jax", "tensorflow").
      strict (bool): If True, the engine will return an error if an API
                     cannot be mapped. If False (default), the original code
                     is preserved wrapped in escape hatch comments.
      plugin_settings (dict, optional): Specific configuration flags passed to
          plugin hooks (e.g., `{"rng_arg_name": "seed"}`).
      semantics (SemanticsManager, optional): An existing Knowledge Base instance.
          If None, a new one is initialized from disk.

  Returns:
      str: The transpiled source code.

  Raises:
      ValueError: If the conversion fails (e.g. syntax errors or strict mode violations).
  """
  # 1. Configure
  config = RuntimeConfig(
    source_framework=source,
    target_framework=target,
    strict_mode=strict,
    plugin_settings=plugin_settings or {},
  )

  # 2. Initialize Engine
  # Note: SemanticsManager loads the knowledge base from JSONs on init.
  manager = semantics or SemanticsManager()
  engine = ASTEngine(semantics=manager, config=config)

  # 3. Execute
  result = engine.run(code)

  # 4. Handle Result
  if not result.success:
    error_msg = "\n".join(result.errors)
    raise ValueError(f"Transpilation failed:\n{error_msg}")

  return result.code


__all__ = [
  "convert",
  "__version__",
]
