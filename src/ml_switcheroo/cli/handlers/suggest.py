"""
Suggest Command Handlers.

This module implements the `suggest` command, which generates a context-rich
prompt for Large Language Models (LLMs). The prompt includes:
1.  Introspection data for the requested API (signatures, docstrings).
2.  The ODL JSON Schema structure.
3.  A Few-Shot example of a correct ODL definition.

This output is designed to be piped directly to an LLM to generate valid
Operation Definition Language (YAML) for missing operations.
"""

import json
import importlib
import inspect
from typing import Optional, Dict, Any

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.utils.console import log_error


def handle_suggest(api_path: str) -> int:
  """
  Generates an LLM prompt for defining a new operation.

  Steps:
  1.  Inspects the live Python object at `api_path` to get signatures and docs.
  2.  Retrieves the JSON Schema for `OperationDef`.
  3.  Constructs a structured prompt text.
  4.  Prints the prompt to stdout.

  Args:
      api_path: The fully qualified name of the python object (e.g. 'torch.nn.Linear').

  Returns:
      int: Exit code (0 for success, 1 for failure).
  """
  # 1. Introspect API
  try:
    info = _inspect_live_object(api_path)
  except (ImportError, AttributeError) as e:
    log_error(f"Could not inspect '{api_path}': {e}. Is the library installed?")
    return 1

  # 2. Get Schema
  schema = json.dumps(OperationDef.model_json_schema(), indent=2)

  # 3. Construct Prompt
  prompt = _build_prompt(api_path, info, schema)

  print(prompt)
  return 0


def _inspect_live_object(api_path: str) -> Dict[str, Any]:
  """
  Locates and inspects a python object by path.

  Args:
      api_path: Dotted string path (e.g. 'torch.nn.Linear').

  Returns:
      Dict containing 'signature', 'docstring', and 'kind'.
  """
  if "." not in api_path:
    raise ImportError(f"Invalid path format: {api_path}")

  module_name, obj_name = api_path.rsplit(".", 1)
  module = importlib.import_module(module_name)
  obj = getattr(module, obj_name)

  doc = inspect.getdoc(obj) or "No documentation available."
  kind = "class" if inspect.isclass(obj) else "function"
  sig = "Unknown Signature"

  try:
    sig = str(inspect.signature(obj))
  except (ValueError, TypeError):
    # Fallback for C-extensions
    pass

  return {
    "signature": sig,
    "docstring": doc,
    "kind": kind,
  }


def _build_prompt(api_path: str, info: Dict[str, Any], schema_json: str) -> str:
  """
  Formats the final prompt template.
  """
  op_name = api_path.split(".")[-1]

  template = f"""You are an expert AI assistant for the 'ml-switcheroo' transpiler project. 
Your task is to generate a valid YAML definition using the Operation Definition Language (ODL). 

--- TARGET OPERATION --- 
Name: {api_path} 
Type: {info["kind"]} 
Signature: {op_name}{info["signature"]} 

Docstring: 

{"\n>".join(info["docstring"].split("\n"))} 

--- OUTPUT FORMAT (JSON SCHEMA) --- 
The YAML must conform rigidly to this Pydantic schema: 
```json
{schema_json} 
```

--- ONE-SHOT EXAMPLE ---
Here is a valid example mapping `torch.abs` to all supported frameworks:
```yml
operation: "Abs"
description: "Calculates the absolute value element-wise."
std_args:
  - name: "x"
    type: "Tensor"
variants:
  torch:
    api: "torch.abs"
  jax:
    api: "jax.numpy.abs"
  flax_nnx:
    api: "jax.numpy.abs" # Flax NNX uses JAX for math
  paxml:
    api: "jax.numpy.abs" # Pax uses JAX for math
  keras:
    api: "keras.ops.abs" # Keras 3 backend-agnostic ops
  tensorflow:
    api: "tf.abs"
  numpy:
    api: "numpy.abs"
  mlx:
    api: "mlx.core.abs"
```
--- INSTRUCTIONS --- 
1. Analyze the Target Operation above. 
2. Define standard arguments (`std_args`) that abstract the core inputs. 
3. Define the `variants` block mapping the source framework ('{api_path.split(".")[0]}') and at least one target (e.g., 'jax' or 'numpy'). 
4. Return ONLY the valid YAML block. 
"""
  return template
