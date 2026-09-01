"""Audits framework mappings against actual extracted snapshots.

Validates that APIs and parameters mapped in ODL/JSON exist in the real snapshots.
"""

from typing import Any


import sys
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Set

# Load local components
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


def extract_api_calls(file_path: Path) -> Set[str]:
  """Extracts fully qualified API calls from a Python file."""
  with open(file_path, "r", encoding="utf-8") as f:
    try:
      tree = ast.parse(f.read())
    except SyntaxError:
      return set()

  aliases: Dict[str, str] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for name in node.names:
        aliases[name.asname or name.name] = name.name
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        for name in node.names:
          aliases[name.asname or name.name] = f"{node.module}.{name.name}"

  apis: Set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      if isinstance(node.func, ast.Attribute):
        chain: List[str] = []
        curr: ast.expr = node.func
        while isinstance(curr, ast.Attribute):
          chain.append(curr.attr)
          curr = curr.value
        if isinstance(curr, ast.Name):
          chain.append(curr.id)
          chain.reverse()
          root = chain[0]
          if root in aliases:
            full_api = aliases[root] + "." + ".".join(chain[1:])
            apis.add(full_api)
  return apis


def audit_inline_snippets(manager: SemanticsManager, snapshots: Dict[str, Dict[str, Any]]) -> List[str]:
  """Audits programmatic API calls inside declarative inline snippets (e.g. macro_template)."""
  errors: List[str] = []

  framework_prefixes = {
    "mlx": "mlx",
    "torch": "torch",
    "jax": "jax",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "stablehlo": "stablehlo",
    "rdna": "rdna",
    "numpy": "numpy",
    "flax": "flax",
    "keras": "keras",
    "praxis": "paxml",
  }

  ignore_list = {
    "jax.image.resize",
    "numpy.add",
    "tf.math.add",
    "jax.numpy.add",
    "keras.ops.add",
    "jax.numpy.sum",
    "jax.numpy.matmul",
    "jax.numpy.linalg.norm",
    "jax.numpy.issubdtype",
    "jax.scipy.special.gammaln",
    "jax.numpy.log",
    "jax.numpy.cumsum",
    "jax.numpy.exp",
    "jax.numpy.log1p",
    "jax.numpy.square",
    "jax.numpy.maximum",
    "jax.numpy.var",
    "jax.lax.complex",
    "jax.numpy.stack",
    "jax.numpy.real",
    "jax.numpy.imag",
    "keras.ops.where",
    "jax.numpy.linalg.slogdet",
    "numpy.generic",
    "numpy.ndarray",
    "torch.tensor",
    "torch.from_numpy",
    "jax.numpy.array",
    "mlx.core.array",
    "tensorflow.convert_to_tensor",
    "keras.ops.convert_to_tensor",
    "jax.default_backend",
    "jax.vjp",
    "jax.random.permutation",
    "jax.random.randint",
    "jax.random.uniform",
    "math.sqrt",
  }

  for op_name, op_details in manager.data.items():
    variants = op_details.get("variants", {})
    for fw_name, fw_mapping in variants.items():
      if "macro_template" in fw_mapping:
        snippet = fw_mapping["macro_template"]
        try:
          tree = ast.parse(snippet)
        except Exception:
          continue

        for node in ast.walk(tree):
          if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
              chain: List[str] = []
              curr: ast.expr = node.func
              while isinstance(curr, ast.Attribute):
                chain.append(curr.attr)
                curr = curr.value
              if isinstance(curr, ast.Name):
                chain.append(curr.id)
                chain.reverse()
                root = chain[0]
                full_api = root + "." + ".".join(chain[1:])

                if full_api in ignore_list:
                  continue

                if root in framework_prefixes:
                  fw_key = framework_prefixes[root]
                  if fw_key in snapshots:
                    snapshot = snapshots[fw_key]
                    if full_api not in snapshot:
                      errors.append(f"[{fw_key}] Inline snippet hallucinated API in {op_name}: '{full_api}'")

  return errors


def audit_python_ast(src_dirs: List[Path], snapshots: Dict[str, Dict[str, Any]]) -> List[str]:
  """Audits programmatic API calls in Python files against snapshots."""
  errors: List[str] = []

  framework_prefixes = {
    "mlx": "mlx",
    "torch": "torch",
    "jax": "jax",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "stablehlo": "stablehlo",
    "rdna": "rdna",
    "numpy": "numpy",
    "flax": "flax",
    "keras": "keras",
    "praxis": "paxml",
  }

  ignore_list = {
    "numpy.generic",
    "numpy.ndarray",
    "torch.tensor",
    "torch.from_numpy",
    "jax.numpy.array",
    "mlx.core.array",
    "tensorflow.convert_to_tensor",
    "keras.ops.convert_to_tensor",
  }

  for src_dir in src_dirs:
    if not src_dir.exists():
      continue
    for file_path in src_dir.rglob("*.py"):
      calls = extract_api_calls(file_path)
      for call in calls:
        if call in ignore_list:
          continue

        root_module = call.split(".")[0]
        if root_module in framework_prefixes:
          fw_key = framework_prefixes[root_module]
          if fw_key in snapshots:
            snapshot = snapshots[fw_key]
            if call not in snapshot:
              errors.append(f"[{fw_key}] Programmatic API call hallucinated in {file_path}: '{call}'")

  return errors


def load_snapshots(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json`.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx') to the JSON data.
  """
  import json

  snapshots: dict[Any, Any] = {}
  for file_path in snapshot_dir.glob("*_v*.json"):
    if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
      continue
    # Extract framework prefix (e.g., 'torch' from 'torch_v2.10.0.json')
    fw = file_path.name.split("_v")[0]
    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      if fw not in snapshots or len(str(snap)) > len(str(snapshots[fw])):
        snapshots[fw] = snap

  flat_snapshots: dict[Any, Any] = {}
  for fw, snap in snapshots.items():
    flat_snapshots[fw] = {}
    for cat, items in snap.get("categories", {}).items():
      if isinstance(items, list):
        for item in items:
          if "api_path" in item:
            flat_snapshots[fw][item["api_path"]] = item
          if "name" in item:
            flat_snapshots[fw][item["name"]] = item
          if "aliases" in item and isinstance(item["aliases"], list):
            for alias in item["aliases"]:
              flat_snapshots[fw][alias] = item
      elif isinstance(items, dict):
        for k, v in items.items():
          flat_snapshots[fw][k] = v
    for k, v in snap.get("functions", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.get("classes", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.items():
      if k not in ["categories", "functions", "classes", "version", "mappings", "templates", "imports", "structs"]:
        if isinstance(v, dict) and "args" in v:
          flat_snapshots[fw][k] = v

  return flat_snapshots


def load_snapshots_multi(snapshot_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
  """Load API snapshots from multiple directories."""
  import json

  snapshots: dict[Any, Any] = {}
  for snapshot_dir in snapshot_dirs:
    if not snapshot_dir.exists():
      continue
    for file_path in snapshot_dir.glob("*_v*.json"):
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      fw = file_path.name.split("_v")[0]
      with open(file_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
        # If we already have a snapshot, we could merge or take latest, but for now just assign
        # (Assuming the globs are sorted or we just want any valid one)
        # To be safe, keep the largest dictionary.
        if fw not in snapshots or len(str(snap)) > len(str(snapshots[fw])):
          snapshots[fw] = snap

  flat_snapshots: dict[Any, Any] = {}
  for fw, snap in snapshots.items():
    flat_snapshots[fw] = {}
    for cat, items in snap.get("categories", {}).items():
      if isinstance(items, list):
        for item in items:
          if "api_path" in item:
            flat_snapshots[fw][item["api_path"]] = item
          if "name" in item:
            flat_snapshots[fw][item["name"]] = item
          if "aliases" in item and isinstance(item["aliases"], list):
            for alias in item["aliases"]:
              flat_snapshots[fw][alias] = item
      elif isinstance(items, dict):
        for k, v in items.items():
          flat_snapshots[fw][k] = v
    for k, v in snap.get("functions", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.get("classes", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.items():
      if k not in ["categories", "functions", "classes", "version", "mappings", "templates", "imports", "structs"]:
        if isinstance(v, dict) and "args" in v:
          flat_snapshots[fw][k] = v
  return flat_snapshots


def audit_frameworks(manager: SemanticsManager, snapshots: Dict[str, Dict[str, Any]]) -> List[str]:
  """Audits the known manager data against the snapshots.

  Args:
      manager: The hydrated SemanticsManager.
      snapshots: The loaded snapshots.

  Returns:
      A list of error strings.
  """
  errors: List[str] = []

  # Ignore known gaps in our automated snapshot extraction (e.g., C-extensions, aliases)
  ignore_list = {
    "numpy.abs",
    "tf.abs",
    "keras.ops.abs",
    "numpy.add",
    "tf.math.add",
    "keras.ops.add",
    "torch.flatten",
    "jax.lax.collapse",
    "numpy.reshape",
    "mlx.core.flatten",
    "numpy.float32",
    "tf.data.Dataset.from_tensor_slices",
    "numpy.mean",
    "tf.math.reduce_mean",
    "keras.ops.mean",
    "numpy.ma.core.multiply",
    "tf.multiply",
    "keras.layers.multiply",
    "keras.random.SeedGenerator",
    "load",
    "save",
    "tf.transpose",
    "numpy.transpose",
    "torch.relu",
    "torch.nn.functional.relu",
  }

  for op_name, op_details in manager.data.items():
    variants = op_details.get("variants", {})
    for fw_name, fw_mapping in variants.items():
      if fw_name not in snapshots:
        # We might not have a snapshot for every framework in the matrix
        continue

      snapshot = snapshots[fw_name]
      api = fw_mapping.get("api")
      if not api:
        continue

      if api not in snapshot:
        # The API is not in our ground-truth snapshot.
        # Ensure the framework is one of our strictly-checked ones to avoid noise from unsupported/partial frameworks.
        if fw_name in ["mlx", "torch", "jax", "tensorflow", "stablehlo", "rdna", "numpy", "flax", "keras"]:
          if api not in ignore_list and not str(api).startswith(";"):
            errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated API: '{api}'")
        continue

      api_data = snapshot[api]

      # Check arguments
      args_map = fw_mapping.get("args", {})
      if not args_map:
        continue

      snapshot_args = api_data.get("params", api_data.get("args", []))
      snapshot_arg_names = {arg["name"] for arg in snapshot_args}

      for std_name, fw_arg_name in args_map.items():
        if fw_arg_name not in snapshot_arg_names:
          # some args might be variadic or **kwargs, but we should verify exact matches if possible
          # check if the api has **kwargs
          has_kwargs = any(arg["name"] == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)
          if not has_kwargs:
            if fw_name in ["mlx", "torch", "jax", "tensorflow", "stablehlo", "rdna", "numpy", "flax", "keras"]:
              errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated argument: '{fw_arg_name}' for API '{api}'")

      # Arity check: Ensure all required target arguments are provided
      if "macro_template" not in fw_mapping and fw_name in [
        "mlx",
        "torch",
        "jax",
        "tensorflow",
        "stablehlo",
        "rdna",
        "numpy",
        "flax",
        "keras",
      ]:
        required_args = {
          arg["name"]
          for arg in snapshot_args
          if arg.get("default") is None
          and arg.get("kind") not in ("VAR_POSITIONAL", "VAR_KEYWORD")
          and arg.get("name") not in ("self", "cls")
        }
        has_kwargs = any(arg["name"] == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)

        mapped_target_args = set(args_map.values())
        missing_args = required_args - mapped_target_args

        # Snapshots often miss defaults for kwargs, base classes, or C-extensions.
        known_missing_defaults = {
          "transposed",
          "output_padding",
          "params",
          "learning_rate",
          "inplace",
          "weight",
          "_modules",
          "num_chunks",
          "name",
          "weight_ih",
          "weight_hh",
          "bias_v",
          "bias_k",
          "values",
          "targets",
          "inputs",
          "predictions",
          "inputs1",
          "inputs2",
          "vars",
          "x1",
          "x2",
          "kernel_size",
          "in_channels",
          "out_channels",
          "num_features",
          "dims",
          "normalized_shape",
          "num_embeddings",
          "num_heads",
          "x",
          "logits",
          "freeze",
          "out_features",
          "input_size",
        }
        missing_args = missing_args - known_missing_defaults

        if missing_args and not has_kwargs:
          # Ignore known gaps
          if api not in ignore_list:
            errors.append(f"[{fw_name}] '{op_name}' missing required arguments for API '{api}': {missing_args}")

  return errors


def main() -> int:
  """Main execution function.

  Returns:
      Exit code (0 for success, 1 for failures).
  """
  parser = argparse.ArgumentParser(description="Audit against snapshots")
  parser.add_argument("--strict", action="store_true", help="Fail if any mismatches found")
  args = parser.parse_args()

  mgr = SemanticsManager()
  KnowledgeBaseLoader(mgr).load_knowledge_graph()
  RegistryLoader(mgr).hydrate()

  snapshot_dirs = [
    Path("../ml-compiler-snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)

  print(f"Loaded {len(snapshots)} snapshots.")
  errors = audit_frameworks(mgr, snapshots)

  src_dirs = [Path("src/ml_switcheroo/frameworks"), Path("src/ml_switcheroo/plugins")]
  ast_errors = audit_python_ast(src_dirs, snapshots)
  errors.extend(ast_errors)

  snippet_errors = audit_inline_snippets(mgr, snapshots)
  errors.extend(snippet_errors)

  if errors:
    print(f"\n❌ Found {len(errors)} mismatches:\n")
    for error in errors:
      print(f"  - {error}")
    if args.strict:
      return 1
  else:
    print("\n✅ All mapped APIs and arguments verified against snapshots.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
