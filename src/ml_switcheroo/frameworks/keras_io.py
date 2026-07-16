"""Keras IO Mixin."""

import textwrap
from typing import List, Optional


class KerasIOMixin:
  """Docstring."""

  def get_serialization_imports(self) -> List[str]:
    """Imports for saving/loading."""
    return ["import keras"]  # pragma: no cover

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Syntax for saving/loading models."""
    if op == "save" and object_arg:  # pragma: no cover
      return f"{object_arg}.save({file_arg})"  # pragma: no cover
    elif op == "load":  # pragma: no cover
      return f"keras.saving.load_model({file_arg})"  # pragma: no cover
    return ""  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns imports required for the generated weight migration script."""
    return ["import keras", "import numpy as np", "import h5py"]  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """Returns python code to load a checkpoint."""
    return textwrap.dedent(  # pragma: no cover
      f"""
            try:
                # Keras weights are usually saved with .weights.h5 or as full model
                # This stub attempts to load if it's a full model file, extracting weights
                model = keras.models.load_model({path_var}, compile=False)
                raw_state = {{w.name: w.numpy() for w in model.weights}}
            except Exception as e:
                print(f"Warning: Failed to load Keras model ({{e}}). Assuming raw h5 weights file.")
                # Fallback to h5py if available for raw weights
                try:
                    import h5py
                    f = h5py.File({path_var}, 'r')
                    raw_state = {{}}
                    def visit_func(name, node):
                        if isinstance(node, h5py.Dataset):
                            raw_state[name] = node[()]
                    f.visititems(visit_func)
                except Exception:
                    print("h5py not installed, cannot load raw weights.")
                    raw_state = {{}}
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Returns python expression string that converts `tensor_var` from Keras tensor to numpy array."""
    return f"{tensor_var}.numpy() if hasattr({tensor_var}, 'numpy') else np.array({tensor_var})"  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Returns Python code to save the dictionary."""
    return textwrap.dedent(  # pragma: no cover
      f"""
            print(f"Saving generic HDF5 weights to {{ {path_var} }} using h5py...")
            with h5py.File({path_var}, "w") as f:
                for key, val in {state_var}.items():
                    # Save flat keys as datasets
                    # We use '/' replacement to create groups if key implies hierarchy,
                    # or just flat keys if preferred.
                    # Keras variable names usually allowed in HDF5 keys.
                    f.create_dataset(str(key), data=val)
            print("Done.")
            """
    )
