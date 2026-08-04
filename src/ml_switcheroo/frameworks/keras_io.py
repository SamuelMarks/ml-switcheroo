"""Keras IO Mixin and serialization helpers.

This module provides the `KerasIOMixin` class, which implements helper methods
for Keras-specific serialization, checkpoint handling, and weight migrations.
These methods facilitate generating Keras-compatible Python scripts for saving/loading
models and converting model weights between Keras and intermediate representations.
"""

import textwrap
from typing import List, Optional


class KerasIOMixin:
  """Mixin class providing Keras-specific IO, weight loading, and serialization capabilities.

  This class implements helper methods for generating import statements,
  handling model saving and loading syntax, and loading/saving checkpoint weights
  specifically tailored for Keras (v3+) compatible models and HDF5 weight files.
  """

  def get_serialization_imports(self) -> List[str]:
    """Returns Python import statements required for Keras serialization operations.

    Returns:
        List[str]: A list of Python import statements needed to serialize or
            deserialize Keras models.
    """
    return ["import keras"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates Keras-specific syntax for model save and load operations.

    Args:
        op (str): The operation to perform, either "save" or "load".
        file_arg (str): The name or path of the target file as a string expression.
        object_arg (Optional[str]): The name of the Keras object to save.
            Required if op is "save". Defaults to None.

    Returns:
        str: A Python code snippet representing the model.save or keras.saving.load_model
            operation, or an empty string if the operation is unsupported or invalid.
    """
    if op == "save" and object_arg:
      return f"{object_arg}.save({file_arg})"
    elif op == "load":
      return f"keras.saving.load_model({file_arg})"
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns Python imports required for Keras weight migration and conversion scripts.

    Returns:
        List[str]: A list of import statements necessary for executing weight
            conversion or checkpoint loading/saving logic.
    """
    return ["import keras", "import numpy as np", "import h5py"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Generates Keras-specific Python code to load checkpoint weights into a dictionary.

    The generated code attempts to load the checkpoint as a compiled/uncompiled Keras
    model first. If that fails, it falls back to raw HDF5 weight extraction via h5py.

    Args:
        path_var (str): The variable name representing the checkpoint file path.

    Returns:
        str: A multi-line Python code snippet representing Keras checkpoint/weights
            loading logic.
    """
    return textwrap.dedent(
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
    """Generates the expression for converting a Keras tensor to a NumPy array.

    Args:
        tensor_var (str): The name of the Keras/TensorFlow tensor variable.

    Returns:
        str: A Python code expression that converts the specified tensor variable
            to a NumPy ndarray.
    """
    return f"{tensor_var}.numpy() if hasattr({tensor_var}, 'numpy') else np.array({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generates Python code to save state dictionary weights to a Keras-compatible HDF5 file.

    Args:
        state_var (str): The name of the variable holding the dictionary of weights
            (keys as weight names, values as NumPy arrays).
        path_var (str): The variable name representing the destination file path.

    Returns:
        str: A multi-line Python code snippet that writes weight dictionary datasets
            into an HDF5 container.
    """
    return textwrap.dedent(
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
