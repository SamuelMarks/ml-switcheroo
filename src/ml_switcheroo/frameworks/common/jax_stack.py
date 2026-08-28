"""JAX Stack Common Logic (Level 0 & Level 1).

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

import textwrap
from typing import Dict, List, Optional


class JAXStackMixin:
  """Mixin providing shared implementations for JAX ecosystem adapters.

  This ensures consistent translation for:

  - Optimization (Torch Optimizers -> Optax Factory Functions).
  - Serialization (Torch Save/Load -> Orbax Checkpointing).
  - Device Management (Torch Device -> JAX Devices).
  - **Test Configuration** (Gen-Tests templates).
  - **Verification Normalization** (JAX Array -> NumPy).
  - **Weight Migration** (Loading/Saving checkpoints via Orbax).
  """

  # --- Test Configuration (Shared) ---

  @property
  def jax_test_config(self) -> Dict[str, str]:
    """Return standard JAX test generation templates using JIT wrapping.

    Defines:
    - `import`: Libraries to import (including opt-in Chex support).
    - `convert_input`: Syntax to convert NumPy array to JAX array.
    - `to_numpy`: Identity transform (preserves PyTrees for Chex comparison).
    - `jit_template`: Detailed JAX JIT syntax with static argument support.

    Returns:
        Dict[str, str]: A dictionary mapping configuration keys to syntax templates.
    """
    return {
      "import": "import jax\nimport jax.numpy as jnp\ntry:\n    import chex\nexcept ImportError:\n    pass",
      "convert_input": "jnp.array({np_var})",
      "to_numpy": "{res_var}",
      "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
    }

  def get_to_numpy_code(self) -> str:
    """Return logic to convert JAX arrays to NumPy.

    Checks for `__array__` protocol which JAX arrays implement.

    Returns:
        str: Python code string representing the conversion logic.
    """
    return "if hasattr(obj, '__array__'): return np.array(obj)"

  # --- Hardware Abstraction (Level 0) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Return JAX-compliant syntax for device specification.

    Maps 'cuda'/'gpu' to 'gpu' backend.
    Maps 'cpu' to 'cpu' backend.

    Args:
        device_type: String literal or variable representing device type (e.g., "'cuda'").
        device_index: Optional index string (e.g., "0").

    Returns:
        str: Python code string constructing the device object: ``jax.devices('gpu')[0]``.
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
    """Return JAX syntax for checking if GPUs are available.

    Format: ``len(jax.devices('gpu')) > 0``.

    Returns:
        str: Python code string representing the GPU check syntax.
    """
    return "len(jax.devices('gpu')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Return JAX syntax for splitting a PRNG key.

    Format: ``rng, key = jax.random.split(rng)``.

    Args:
        rng_var: Variable name for the input and output PRNG key.
        key_var: Variable name for the split-off child key.

    Returns:
        str: Python code string representing the split operation.
    """
    return f"{rng_var}, {key_var} = jax.random.split({rng_var})"

  # --- IO Serialization (Level 1 - Orbax) ---

  def get_serialization_imports(self) -> List[str]:
    """Return standard imports for JAX serialization via Orbax.

    Returns:
        List[str]: List of standard import statement strings.
    """
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Return Orbax syntax for save/load operations.

    Args:
        op: Operation name ('save' or 'load').
        file_arg: Path to checkpoint directory.
        object_arg: The PyTree to save (required for save).

    Returns:
        str: Python code string.
    """
    if op == "save" and object_arg:
      return f"orbax.checkpoint.PyTreeCheckpointer().save(directory={file_arg}, item={object_arg})"
    elif op == "load":
      return f"orbax.checkpoint.PyTreeCheckpointer().restore({file_arg})"
    return ""

  # --- Weight Migration (Adapter) ---

  def get_weight_conversion_imports(self) -> List[str]:
    """Return imports required for the generated weight migration script.

    Returns:
        List[str]: List of import statements.
    """
    return [
      "import jax.numpy as jnp",
      "import orbax.checkpoint",
      "from flax.traverse_util import unflatten_dict, flatten_dict",
      "try:",
      "    import safetensors.flax",
      "except ImportError:",
      "    pass",
    ]

  def get_weight_load_code(self, path_var: str) -> str:
    """Return python code to load a checkpoint from `path_var` into a variable named `raw_state`.

    The `raw_state` is a flat dictionary where keys are dot-separated strings (e.g. 'layer.weight').

    Args:
        path_var: Variable name containing the directory path to load.

    Returns:
        str: Python code snippet representing weight loading logic.
    """
    return textwrap.dedent(
      f"""
            if str({path_var}).endswith(".safetensors"):
                raw_tree = safetensors.flax.load_file({path_var})
            else:
                # Load with Orbax and Flatten
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                raw_tree = checkpointer.restore({path_var})

            if 'params' in raw_tree:
                raw_tree = raw_tree['params']

            # Helper to flatten with tuple keys
            # Key format: (layer_name, param_name) tuple
            # We convert tuple keys to dot-separated strings for the interop mapping
            raw_state = {{
                ".".join(k) : v for k, v in flatten_dict(raw_tree).items()
            }}
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Return a python expression string that converts `tensor_var` from JAX array to NumPy array.

    Args:
        tensor_var: Variable name of the JAX array.

    Returns:
        str: Python expression string.
    """
    return f"np.array({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Return python code to save the dictionary `state_var` (mapping flat keys to NumPy arrays).

    to `path_var`. It unstricts flat keys back to PyTree structure using `unflatten_dict` and saves via Orbax or safetensors.

    Args:
        state_var: Name of the dictionary mapping flat keys to NumPy arrays.
        path_var: Target path variable.

    Returns:
        str: Python code snippet representing weight saving logic.
    """
    return textwrap.dedent(
      f"""
            # Restructure PyTree
            # Convert dot keys back to tuple keys
            tuple_params = {{tuple(k.split('.')): v for k, v in {state_var}.items()}}
            params_tree = unflatten_dict(tuple_params)

            if str({path_var}).endswith(".safetensors"):
                safetensors.flax.save_file(params_tree, {path_var})
            else:
                final_tree = {{'params': params_tree}}
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpointer.save({path_var}, final_tree)
            """
    )

  # --- Documentation Linking ---

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Generate a default documentation URL for standard JAX APIs.

    Maps to ReadTheDocs autosummary path.
    NOTE: Subclasses (Flax/Pax) should override this for their specific namespaces.

    Args:
        api_name: The fully qualified API path (e.g. 'jax.numpy.abs').

    Returns:
        Optional[str]: String URL or None.
    """
    return f"https://jax.readthedocs.io/en/latest/_autosummary/{api_name}.html"

  # --- Manual Wiring (Semantics Injection / Legacy Support) ---

  def _apply_stack_wiring(self, snapshot) -> None:
    """Inject mappings common to all JAX frameworks (JNP, Optax, JIT).

    This method populates the semantic snapshot with rules for translating
    Torch/NumPy concepts to the JAX ecosystem equivalents.

    NOTE: This is largely superseded by the static `definitions` property on the Adapter,
    but preserved for dynamic wiring use cases (e.g. PaxML manual overlays).

    Args:
        snapshot: The semantic snapshot dictionary to mutate.
                  Expected structure: {'mappings': {}, 'templates': {}}

    Returns:
        None (mutates the snapshot dictionary in-place).
    """
    mappings = snapshot.setdefault("mappings", {})
    templates = snapshot.setdefault("templates", {})

    if not templates:
      templates.update(self.jax_test_config)

    # 1. Core JAX Operation rewrites (Level 0)
    # mappings["Abs"] = {"api": "jax.numpy.abs"}
    # mappings["abs"] = {"api": "jax.numpy.abs"}
    # mappings["permute_dims"] = {"api": "jax.numpy.transpose", "pack_to_tuple": "axes"}
    # mappings["size"] = {"api": "shape", "requires_plugin": "method_to_property"}
    # mappings["data_ptr"] = {"api": "data", "requires_plugin": "method_to_property"}
    # mappings["Einsum"] = {"api": "jax.numpy.einsum", "requires_plugin": "einsum_normalizer"}

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
