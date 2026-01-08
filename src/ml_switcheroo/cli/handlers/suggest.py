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
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.utils.console import log_error


def handle_suggest(api_path: str, out_dir: Optional[Path] = None, batch_size: int = 50) -> int:
  """
  Generates an LLM prompt for defining new operations.

  Supports both single API paths (e.g. ``torch.nn.Linear``) and
  module wildcards (e.g. ``jax.numpy.*``).

  Steps:
  1.  Resolves target objects (single or list from wildcard).
  2.  Inspects live Python objects to get signatures and docs.
  3.  Retrieves the JSON Schema for ``OperationDef``.
  4.  Constructs structured prompts with Header, Batched Ops, and Footer.
  5.  Writes output to stdout or files if out_dir specified.

  Args:
      api_path: The path to inspect. Can be a dotted path to an object
          or a module path ending in ``.*``.
      out_dir: Optional directory to save batched .md files.
      batch_size: Number of operations per batch/file.

  Returns:
      int: Exit code (0 for success, 1 for failure).
  """
  # 1. Resolve Targets
  targets: List[Tuple[str, Dict[str, Any]]] = []

  if api_path.endswith(".*"):
    module_name = api_path[:-2]
    try:
      module = importlib.import_module(module_name)
      # Scan public members
      for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
          continue
        full_path = f"{module_name}.{name}"
        try:
          # We skip modules/builtins that might clutter unless they look like ops
          if inspect.ismodule(obj):
            continue
          info = _extract_metadata(obj)
          targets.append((full_path, info))
        except Exception:
          continue
    except ImportError as e:
      log_error(f"Could not import module '{module_name}': {e}")
      return 1
  else:
    # Single mode
    try:
      info = _inspect_live_object(api_path)
      targets.append((api_path, info))
    except (ImportError, AttributeError) as e:
      log_error(f"Could not inspect '{api_path}': {e}. Is the library installed?")
      return 1

  if not targets:
    log_error(f"No valid API targets found for '{api_path}'.")
    return 1

  # Sort deterministically
  targets.sort(key=lambda x: x[0])

  # 2. Get Schema
  schema = json.dumps(OperationDef.model_json_schema(), indent=2)

  # Determine base properties for filename generation
  base_name = api_path.replace(".*", "").replace(".", "_")

  # Guess source framework from first target found
  source_fw = targets[0][0].split(".")[0]

  header_text = _build_header(schema)
  footer_text = _build_footer(source_fw)

  # 3. Chunking logic
  chunks = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]

  if out_dir:
    if not out_dir.exists():
      out_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
      content = [header_text]
      for path, info in chunk:
        content.append(_build_target_block(path, info))
      content.append(footer_text)

      filename = out_dir / f"suggest_{base_name}_{i + 1:03d}.md"
      with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
      print(f"Generated {filename}")

  else:
    # Stdout Logic (Same structure, single stream)
    print(header_text)
    for chunk in chunks:
      # Print blocks
      for path, info in chunk:
        print(_build_target_block(path, info))
    print(footer_text)

  return 0


def _extract_metadata(obj: Any) -> Dict[str, Any]:
  """
  Extracts signature and docstring from a live object.

  Args:
      obj: The Python object to inspect.

  Returns:
      Dict: Metadata containing 'signature', 'docstring', and 'kind'.
  """
  doc = inspect.getdoc(obj) or "No documentation available."
  kind = "class" if inspect.isclass(obj) else "function"
  sig = "Unknown Signature"

  try:
    # signature() returns the parameter list e.g. "(x, y=1)"
    sig = str(inspect.signature(obj))
  except (ValueError, TypeError):
    # Fallback for C-extensions
    pass

  return {
    "signature": sig,
    "docstring": doc,
    "kind": kind,
  }


def _inspect_live_object(api_path: str) -> Dict[str, Any]:
  """
  Locates and inspects a python object by path.

  Args:
      api_path: Dotted string path (e.g. 'torch.nn.Linear').

  Returns:
      Dict containing 'signature', 'docstring', and 'kind'.

  Raises:
      ImportError: If module cannot be loaded or path format is invalid.
      AttributeError: If object not found in module.
  """
  if "." not in api_path:
    raise ImportError(f"Invalid path format: {api_path}")

  module_name, obj_name = api_path.rsplit(".", 1)
  module = importlib.import_module(module_name)
  obj = getattr(module, obj_name)

  return _extract_metadata(obj)


def _build_header(schema_json: str) -> str:
  """Returns the static prompt header with Context and Examples."""
  return f"""You are an expert AI assistant for the 'ml-switcheroo' transpiler project. 
Your task is to generate valid YAML definitions using the Operation Definition Language (ODL). 

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
"""


def _build_target_block(api_path: str, info: Dict[str, Any]) -> str:
  """Returns the descriptive block for a single operation."""
  op_name = api_path.split(".")[-1]
  newline = "\n"

  return f"""
--- TARGET OPERATION --- 
Name: {api_path} 
Type: {info["kind"]} 
Signature: {op_name}{info["signature"]} 

Docstring: 

{(newline + ">").join(info["docstring"].split(newline))} 
"""


def _build_footer(source_fw: str) -> str:
  """Returns the final instructions."""
  return f"""
--- INSTRUCTIONS --- 
1. Analyze the Target Operations listed above. 
2. Define standard arguments (`std_args`) that abstract the core inputs. 
3. Define the `variants` block mapping the source framework ('{source_fw}') and at least one target (e.g., 'jax' or 'numpy'). 
4. Return ONLY the valid YAML block(s), separated by '---' if multiple. 
"""
