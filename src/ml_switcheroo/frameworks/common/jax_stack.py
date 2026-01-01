"""
JAX Stack Common Logic (Level 0 & Level 1).

This module provides the ``JAXStackMixin``, a reusable base for any Framework Adapter
built on top of the JAX ecosystem (e.g., Flax, PaxML, Haiku).

It standardizes:

1.  **Level 0 (Core)**: JIT compilation templates, Device syntax (``jax.devices``),
    and Array API mappings (``jax.numpy``).
2.  **Level 1 (Common Libs)**:

    - **Optax**: Optimization primitives and loss functions.
    - **Orbax**: Checkpointing and Serialization.

Usage:
    class MyJaxFramework(JAXStackMixin):
        # ... logic ...
"""

from typing import Dict, Any, List, Optional


class JAXStackMixin:
  """
  Mixin providing shared implementations for JAX ecosystem adapters.

  This ensures consistent translation for:

  - Optimization (Torch Optimizers -> Optax Factory Functions).
  - Serialization (Torch Save/Load -> Orbax Checkpointing).
  - Device Management (Torch Device -> JAX Devices).
  - **Test Configuration** (Gen-Tests templates).
  - **Verification Normalization** (JAX Array -> NumPy).
  """

  # --- Test Configuration (Shared) ---

  @property
  def jax_test_config(self) -> Dict[str, str]:
    """
    Returns standard JAX test generation templates using JIT wrapping.

    Defines:
    - `import`: Libraries to import (including opt-in Chex support).
    - `convert_input`: Syntax to convert Numpy array to JAX array.
    - `to_numpy`: Identity transform (preserves PyTrees for Chex comparison).
    - `jit_template`: Detailed JAX JIT syntax with static argument support.
    """
    return {
      "import": "import jax\nimport jax.numpy as jnp\ntry:\n    import chex\nexcept ImportError:\n    pass",
      "convert_input": "jnp.array({np_var})",
      "to_numpy": "{res_var}",
      "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
    }

  def get_to_numpy_code(self) -> str:
    """
    Returns logic to convert JAX arrays to NumPy.
    Checks for `__array__` protocol which JAX arrays implement.
    """
    return "if hasattr(obj, '__array__'): return np.array(obj)"

  # --- Hardware Abstraction (Level 0) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns JAX-compliant syntax for device specification.

    Maps 'cuda'/'gpu' to 'gpu' backend.
    Maps 'cpu' to 'cpu' backend.

    Args:
        device_type: String literal or variable representing device type (e.g., "'cuda'").
        device_index: Optional index string (e.g., "0").

    Returns:
        Python code string constructing the device object: ``jax.devices('gpu')[0]``.
    """
    # Clean quotes if present to check value
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    # Reconstruct string literal if original was a literal
    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

  def get_device_check_syntax(self) -> str:
    """
    Returns JAX syntax for checking if GPUs are available.
    Format: ``len(jax.devices('gpu')) > 0``
    """
    return "len(jax.devices('gpu')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns JAX syntax for splitting a PRNG key.
    Format: ``rng, key = jax.random.split(rng)``
    """
    return f"{rng_var}, {key_var} = jax.random.split({rng_var})"

  # --- IO Serialization (Level 1 - Orbax) ---

  def get_serialization_imports(self) -> List[str]:
    """Returns standard imports for JAX serialization via Orbax."""
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns Orbax syntax for save/load operations.

    Args:
        op: Operation name ('save' or 'load').
        file_arg: Path to checkpoint directory.
        object_arg: The PyTree to save (required for save).

    Returns:
        Python code string.
    """
    if op == "save" and object_arg:
      return f"orbax.checkpoint.PyTreeCheckpointer().save(directory={file_arg}, item={object_arg})"
    elif op == "load":
      return f"orbax.checkpoint.PyTreeCheckpointer().restore({file_arg})"
    return ""

  # --- Manual Wiring (Semantics Injection / Legacy Support) ---

  def _apply_stack_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Injects mappings common to all JAX frameworks (JNP, Optax, JIT).

    This method populates the semantic snapshot with rules for translating
    Torch/NumPy concepts to the JAX ecosystem equivalents.

    NOTE: This is largely superseded by the static `definitions` property on the Adapter,
    but preserved for dynamic wiring use cases (e.g. PaxML manual overlays).

    Args:
        snapshot: The semantic snapshot dictionary to mutate.
                  Expected structure: {'mappings': {}, 'templates': {}}
    """
    mappings = snapshot.setdefault("mappings", {})
    templates = snapshot.setdefault("templates", {})

    if not templates:
      templates.update(self.jax_test_config)

    # 1. Core JAX Operation rewrites (Level 0)
    mappings["Abs"] = {"api": "jnp.abs"}
    mappings["abs"] = {"api": "jnp.abs"}
    mappings["permute_dims"] = {"api": "jnp.transpose", "pack_to_tuple": "axes"}
    mappings["size"] = {"api": "shape", "requires_plugin": "method_to_property"}
    mappings["data_ptr"] = {"api": "data", "requires_plugin": "method_to_property"}
    mappings["Einsum"] = {"api": "jnp.einsum", "requires_plugin": "einsum_normalizer"}

    # 2. Optax Wiring (Level 1)
    mappings["step"] = {"requires_plugin": "optimizer_step"}
    mappings["zero_grad"] = {"requires_plugin": "optimizer_zero_grad"}
    mappings["Adam"] = {
      "api": "optax.adam",
      "requires_plugin": "optimizer_constructor",
    }

    # 3. Control Flow Templates (Level 0)
    templates["fori_loop"] = "val = jax.lax.fori_loop({start}, {stop}, lambda i, val: {body}, {init_val})"
    templates["scan"] = "carry, stacked = jax.lax.scan(lambda c, x: {body}, {init}, {xs})"
