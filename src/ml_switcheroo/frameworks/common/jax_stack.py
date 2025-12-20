"""
JAX Stack Common Logic (Level 0 & Level 1).

This module provides the `JAXStackMixin`, a reusable base for any Framework Adapter
built on top of the JAX ecosystem (e.g., Flax, PaxML, Haiku).

It standardizes:
1.  **Level 0 (Core)**: JIT compilation templates, Device syntax (`jax.devices`),
    and Array API mappings (`jax.numpy`).
2.  **Level 1 (Common Libs)**:
    - **Optax**: Optimization primitives and loss functions.
    - **Orbax**: Checkpointing and Serialization.

Usage:
    class MyJaxFramework(JAXStackMixin):
        def apply_wiring(self, snapshot):
            self._apply_stack_wiring(snapshot)
            # ... custom logic ...
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
  """

  # --- Test Configuration (Shared) ---

  @property
  def jax_test_config(self) -> Dict[str, str]:
    """
    Returns standard JAX test generation templates using JIT wrapping.
    """
    return {
      "import": "import jax\nimport jax.numpy as jnp",
      "convert_input": "jnp.array({np_var})",
      "to_numpy": "np.array({res_var})",
      "jit_wrap": "True",  # Enable experimental JIT wrapper generation
    }

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
        Python code string constructing the device object: `jax.devices('gpu')[0]`.
    """
    # Clean quotes if present to check value
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    # Reconstruct string literal if original was a literal
    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

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

  # --- Manual Wiring (Semantics Injection) ---

  def _apply_stack_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Injects mappings common to all JAX frameworks (JNP, Optax, JIT).

    This method populates the semantic snapshot with rules for translating
    Torch/NumPy concepts to the JAX ecosystem equivalents.

    Args:
        snapshot: The semantic snapshot dictionary to mutate.
                  Expected structure: {'mappings': {}, 'templates': {}}
    """
    mappings = snapshot.setdefault("mappings", {})
    templates = snapshot.setdefault("templates", {})

    # Inject test templates if not present
    # Usually handled by handle_sync pulling from test_config,
    # but populating here is a safe fallback for direct wiring calls.
    if not templates:
      templates.update(self.jax_test_config)

    # 1. Core JAX Operation rewrites (Level 0)
    # Ensure 'Abs' (Capitalized Abstract) maps to 'jnp.abs'
    mappings["Abs"] = {"api": "jnp.abs"}
    mappings["abs"] = {"api": "jnp.abs"}

    # Handle varargs packing for Permute/Transpose
    mappings["permute_dims"] = {"api": "jnp.transpose", "requires_plugin": "pack_varargs"}

    # Method -> Property swaps common in JAX arrays (e.g. .size() -> .shape)
    mappings["size"] = {"api": "shape", "requires_plugin": "method_to_property"}
    mappings["data_ptr"] = {"api": "data", "requires_plugin": "method_to_property"}

    # Einsum normalization (Equation first)
    mappings["Einsum"] = {"api": "jnp.einsum", "requires_plugin": "einsum_normalizer"}

    # 2. Optax Wiring (Level 1)
    # Maps generic imperative optimizer methods to functional equivalents or plugins
    mappings["step"] = {"requires_plugin": "optimizer_step"}
    mappings["zero_grad"] = {"requires_plugin": "optimizer_zero_grad"}

    # Wire standard configurations for known optimizers
    # PyTorch Optimizers (Class) -> Optax Optimizers (Factory Function)
    for opt_name in ["Adam", "SGD", "RMSprop", "AdamW", "Adagrad", "LBFGS", "Yogi", "AdaBelief"]:
      if opt_name not in mappings:
        mappings[opt_name] = {}

      # The 'optimizer_constructor' plugin strips the 'params' argument
      # because Optax optimizers are initialized stateless (just hyperparameters).
      mappings[opt_name]["requires_plugin"] = "optimizer_constructor"

      # Default API path (e.g., optax.adam)
      if "api" not in mappings[opt_name]:
        mappings[opt_name]["api"] = f"optax.{opt_name.lower()}"

    # 3. Control Flow Templates (Level 0)
    # Defines how loops are generated if an adapter requests structural rewrites
    templates["fori_loop"] = "val = jax.lax.fori_loop({start}, {stop}, lambda i, val: {body}, {init_val})"
    templates["scan"] = "carry, stacked = jax.lax.scan(lambda c, x: {body}, {init}, {xs})"
