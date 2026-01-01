"""
Test Generation Templates and configuration descriptors.

This module stores default code templates for supported frameworks and
provides utilities to determine properties of test arguments (e.g., static JIT args).
"""

from typing import Dict, Any, Optional

#: Default code templates used as fallbacks if the SemanticsManager is unavailable.
DEFAULT_TEST_TEMPLATES = {
  "torch": {
    "import": "import torch",
    "convert_input": "torch.tensor({np_var})",
    "to_numpy": "{res_var}.numpy()",
  },
  "jax": {
    "import": "import jax\nimport jax.numpy as jnp",
    "convert_input": "jnp.array({np_var})",
    "to_numpy": "np.array({res_var})",
    "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
  },
  "tensorflow": {
    "import": "import tensorflow as tf",
    "convert_input": "tf.convert_to_tensor({np_var})",
    "to_numpy": "{res_var}.numpy()",
  },
}


def get_template(manager: Any, framework: str) -> Dict[str, str]:
  """
  Retrieves the code generation template for a specific framework.

  Priority:
  1. SemanticsManager lookup (loaded from snapshots).
  2. Hardcoded defaults in `DEFAULT_TEST_TEMPLATES`.
  3. Empty dict.

  Args:
      manager: The SemanticsManager instance (can be None).
      framework: The framework key (e.g., 'torch', 'jax').

  Returns:
      Dict[str, str]: Template strings for imports, conversion, etc.
  """
  tmpl: Optional[Dict[str, str]] = None
  if manager:
    try:
      tmpl = manager.get_test_template(framework)
    except Exception:
      pass

  if tmpl:
    return tmpl

  return DEFAULT_TEST_TEMPLATES.get(framework, {})


def is_static_arg(arg_info: Dict[str, Any]) -> bool:
  """
  Determines if an argument should be marked static for JIT compilation.

  Heuristic checks for primitive types (int, bool, str) or specific names
  common to axis/dimension arguments.

  Args:
      arg_info: A dictionary containing 'name' and 'type' keys.

  Returns:
      bool: True if the argument should be static.
  """
  t = arg_info.get("type", "")
  # Heuristic: primitives are usually static in JAX JIT context for shapes/axes
  if t.lower() in ("int", "bool", "str", "List[int]", "Tuple[int]"):
    return True
  elif arg_info.get("name") in ("axis", "dim", "keepdims"):
    return True
  else:
    return False
