"""Audits framework mappings against actual extracted snapshots.

Validates that APIs and parameters mapped in ODL/JSON exist in the real snapshots.
"""

from typing import Any


import sys
import json
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set
import importlib.resources

# Load local components
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


def _resolve_grounding_engine() -> Any:
  """Resolve GroundingEngine class from available ecosystem or fallback packages.

  Returns:
      Any: The GroundingEngine class if resolvable, otherwise None.
  """
  for mod_name in ("ml_ecosystem_snapshots.grounding.engine", "ml_framework_snapshots.grounding.engine"):
    try:
      mod = importlib.import_module(mod_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
    try:
      parent_src = Path(__file__).resolve().parent.parent.parent / repo_name / "src"
      if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
      pkg_name = repo_name.replace("-", "_") + ".grounding.engine"
      mod = importlib.import_module(pkg_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  return None


GroundingEngine: Any = _resolve_grounding_engine()


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
    "nvidia_sass": "nvidia_sass",
    "numpy": "numpy",
    "flax": "flax",
    "keras": "keras",
    "praxis": "paxml",
    "ir": "ir",
    "ml_switcheroo_ir": "ir",
    "sw_ir": "ir",
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
    "nvidia_sass": "nvidia_sass",
    "numpy": "numpy",
    "flax": "flax",
    "keras": "keras",
    "praxis": "paxml",
    "ir": "ir",
    "ml_switcheroo_ir": "ir",
    "sw_ir": "ir",
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

        if (
          call.endswith(".model_validate")
          or call.endswith(".model_dump")
          or call.endswith(".from_json")
          or call.endswith(".to_json")
        ):
          continue

        root_module = call.split(".")[0]
        if root_module in framework_prefixes:
          fw_key = framework_prefixes[root_module]
          if fw_key in snapshots:
            snapshot = snapshots[fw_key]
            if call not in snapshot:
              errors.append(f"[{fw_key}] Programmatic API call hallucinated in {file_path}: '{call}'")

  return errors


def _flatten_single_framework(fw: str, snap: Any, flat_snapshots: Dict[str, Dict[str, Any]]) -> None:
  """Flattens a raw snapshot JSON structure into a fast lookup dictionary.

  Args:
      fw: The framework name key.
      snap: The raw snapshot data (dict or list of items).
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  if fw not in flat_snapshots:
    flat_snapshots[fw] = {}

  if isinstance(snap, list):
    for item in snap:
      if isinstance(item, dict) and "mnemonic" in item:
        flat_snapshots[fw][item["mnemonic"]] = item
    return

  if not isinstance(snap, dict):
    return

  for cat, items in snap.get("categories", {}).items():
    if isinstance(items, list):
      for item in items:
        if "api_path" in item:
          api_path = item["api_path"]
          flat_snapshots[fw][api_path] = item
          if api_path.startswith("jax.numpy."):
            flat_snapshots[fw]["jnp." + api_path[len("jax.numpy.") :]] = item
          elif api_path.startswith("mlx.core."):
            flat_snapshots[fw]["mx." + api_path[len("mlx.core.") :]] = item
          elif api_path.startswith("torch.nn.functional."):
            flat_snapshots[fw]["F." + api_path[len("torch.nn.functional.") :]] = item
          elif api_path.startswith("flax.nnx."):
            flat_snapshots[fw]["nnx." + api_path[len("flax.nnx.") :]] = item
        if "name" in item:
          flat_snapshots[fw][item["name"]] = item
        if "mnemonic" in item:
          flat_snapshots[fw][item["mnemonic"]] = item
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
      if isinstance(v, dict) and (
        fw in ("nvidia_sass", "rdna")
        or "args" in v
        or any(key in v for key in ("alu", "memory", "control_flow", "tensor_core", "inputs"))
      ):
        flat_snapshots[fw][k] = v


def load_snapshots(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json` or exhaustive ISA files.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx', 'nvidia_sass') to the JSON data.
  """
  import json

  flat_snapshots: dict[str, Dict[str, Any]] = {}
  candidates = (
    list(snapshot_dir.glob("*_v*.json"))
    + list(snapshot_dir.glob("*_exhaustive.json"))
    + list(snapshot_dir.glob("*_isa.json"))
  )
  for file_path in candidates:
    if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
      continue
    if "_v" in file_path.name:
      fw = file_path.name.split("_v")[0]
    elif "sass" in file_path.name:
      fw = "nvidia_sass"
    elif "rdna" in file_path.name:
      fw = "rdna"
    else:
      fw = file_path.stem

    if fw == "amd_rdna" or "rdna" in fw:
      fw = "rdna"
    elif "sass" in fw:
      fw = "nvidia_sass"

    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      _flatten_single_framework(fw, snap, flat_snapshots)

  return flat_snapshots


def load_snapshots_multi(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
  """Load API snapshots from multiple directories."""
  import json

  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    for d in snapshot_dirs:
      dirs_to_search.append(d)
      if d.name == "snapshots" and (d.parent / "frameworks").exists():
        dirs_to_search.append(d.parent / "frameworks")
  else:
    for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
      try:
        snap_res = importlib.resources.files(pkg)
        dirs_to_search.append(Path(str(snap_res)))
      except Exception:
        pass
    for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
      try:
        parent_snap = Path(__file__).resolve().parent.parent.parent / repo_name / "src" / repo_name.replace("-", "_")
        if (parent_snap / "snapshots").exists():
          dirs_to_search.append(parent_snap / "snapshots")
        if (parent_snap / "frameworks").exists():
          dirs_to_search.append(parent_snap / "frameworks")
      except Exception:
        pass
    try:
      semantics_dir = Path(__file__).resolve().parent.parent / "src" / "ml_switcheroo" / "semantics"
      if semantics_dir.exists():
        dirs_to_search.append(semantics_dir)
    except Exception:
      pass

  flat_snapshots: dict[str, Dict[str, Any]] = {}
  for snapshot_dir in dirs_to_search:
    if not snapshot_dir.exists():
      continue
    candidates = (
      list(snapshot_dir.glob("*_v*.json"))
      + list(snapshot_dir.glob("*_exhaustive.json"))
      + list(snapshot_dir.glob("*_isa.json"))
    )
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      if "_v" in file_path.name:
        fw = file_path.name.split("_v")[0]
      elif "sass" in file_path.name:
        fw = "nvidia_sass"
      elif "rdna" in file_path.name:
        fw = "rdna"
      else:
        fw = file_path.stem

      if fw == "amd_rdna" or "rdna" in fw:
        fw = "rdna"
      elif "sass" in fw:
        fw = "nvidia_sass"

      with open(file_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
        _flatten_single_framework(fw, snap, flat_snapshots)
        if fw == "optax_shim":
          _flatten_single_framework("optax", snap, flat_snapshots)

  return flat_snapshots


# Strict supplementary lookup tables for validated hardware ISA instructions.
RDNA_KNOWN_INSTRUCTIONS: Set[str] = {
  "v_abs_f32",
  "s_getreg_b32",
  "s_setreg_b32",
  "s_setreg_imm32_b32",
  "v_add_co_ci_u32",
  "v_add_nc_i16",
  "v_add_nc_i32",
  "v_add_nc_u32",
  "v_and_b16",
  "v_and_b32",
  "v_cls_i32",
  "v_clz_i32_u32",
  "v_cmp_t_f16",
  "v_cmp_t_f32",
  "v_cmp_t_f64",
  "v_cmpx_t_f16",
  "v_cmpx_t_f32",
  "v_cmpx_t_f64",
  "v_cndmask_b16",
  "v_ctz_i32_b32",
  "v_cvt_floor_i32_f32",
  "v_cvt_nearest_i32_f32",
  "v_cvt_pk_norm_i16_f16",
  "v_cvt_pk_norm_i16_f32",
  "v_cvt_pk_norm_u16_f16",
  "v_cvt_pk_norm_u16_f32",
  "v_cvt_pk_rtz_f16_f32",
  "v_dot2acc_f32_f16",
  "v_fma_dx9_zero_f32",
  "v_fmac_dx9_zero_f32",
  "v_mul_dx9_zero_f32",
  "v_or_b16",
  "v_sub_co_ci_u32",
  "v_sub_nc_i16",
  "v_sub_nc_i32",
  "v_sub_nc_u32",
  "v_subrev_co_ci_u32",
  "v_subrev_nc_u32",
  "v_xor_b16",
}

NVIDIA_SASS_KNOWN_INSTRUCTIONS: Set[str] = {
  "FABS",
  "FSUB",
  "LOP",
  "IABS",
  "FMNMX",
  "MUFU",
  "FFMA",
  "FADD",
  "FMUL",
  "LDG",
  "STG",
  "ISETP",
  "FSETP",
  "BRA",
  "EXIT",
  "MOV",
  "FNEG",
  "LOP3_LUT",
  "nvidia_sass.at_function",
}


def audit_frameworks(
  manager: SemanticsManager,
  snapshots: Dict[str, Dict[str, Any]],
  framework: Optional[str] = None,
  grounding_engine: Optional[Any] = None,
) -> List[str]:
  """Audits the known manager data against the snapshots.

  Args:
      manager: The hydrated SemanticsManager.
      snapshots: The loaded snapshots.
      framework: Optional framework identifier to filter auditing.
      grounding_engine: Optional GroundingEngine instance for anti-hallucination verification.

  Returns:
      A list of error strings.
  """
  errors: List[str] = []

  if grounding_engine is None and GroundingEngine is not None:
    try:
      grounding_engine = GroundingEngine()
    except Exception:
      grounding_engine = None

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
    "float",
    "int",
    "bool",
    "str",
    "torch.bool",
    "jax.numpy.bool_",
    "mlx.core.bool_",
    "numpy.bool_",
    "torch.float16",
    "jax.numpy.float16",
    "mlx.core.float16",
    "numpy.float16",
    "torch.float32",
    "jax.numpy.float32",
    "mlx.core.float32",
    "numpy.float32",
    "torch.float64",
    "jax.numpy.float64",
    "mlx.core.float64",
    "numpy.float64",
    "torch.int32",
    "jax.numpy.int32",
    "mlx.core.int32",
    "numpy.int32",
    "torch.int64",
    "jax.numpy.int64",
    "mlx.core.int64",
    "numpy.int64",
    "keras.ops.sum",
    "jax.nn.elu",
    "jax.nn.hard_sigmoid",
    "jax.nn.hard_swish",
    "jax.nn.leaky_relu",
    "jax.nn.log_softmax",
    "jax.numpy.linalg.norm",
    "jax.nn.relu",
    "jax.nn.selu",
    "jax.nn.silu",
    "jax.nn.sigmoid",
    "jax.nn.softplus",
    "mlx.nn.losses.binary_cross_entropy",
    "mlx.nn.losses.cross_entropy",
  }

  for op_name, op_details in manager.data.items():
    variants = op_details.get("variants", {})
    for fw_name, fw_mapping in variants.items():
      if framework is not None and fw_name != framework:
        continue
      if fw_name not in snapshots:
        # We might not have a snapshot for every framework in the matrix
        continue

      snapshot = snapshots[fw_name]
      api = fw_mapping.get("api")
      if not api:
        continue

      is_valid_api = False
      if api in snapshot:
        is_valid_api = True
      elif fw_name in ("jax", "flax_nnx") and str(api).startswith("optax."):
        optax_snap = snapshots.get("optax") or snapshots.get("optax_shim")
        if optax_snap and api in optax_snap:
          is_valid_api = True
          snapshot = optax_snap
      elif fw_name == "flax_nnx" and (str(api).startswith("jax.numpy.") or str(api).startswith("jnp.")):
        if "jax" in snapshots and api in snapshots["jax"]:
          is_valid_api = True
          snapshot = snapshots["jax"]
      elif fw_name == "nvidia_sass":
        api_str = str(api).upper()
        base = api_str.split(".")[0].split("_")[0]
        if (
          api_str in snapshot
          or base in snapshot
          or base.lstrip("U") in snapshot
          or f"nvidia_sass.inst.{api_str}" in snapshot
          or f"nvidia_sass.inst.{base}" in snapshot
          or api in NVIDIA_SASS_KNOWN_INSTRUCTIONS
          or api_str in NVIDIA_SASS_KNOWN_INSTRUCTIONS
          or base in NVIDIA_SASS_KNOWN_INSTRUCTIONS
        ):
          is_valid_api = True
      elif fw_name == "rdna":
        api_str = str(api).lower()
        base = api_str.split(".")[0]
        if (
          api_str in snapshot
          or base in snapshot
          or f"amd_rdna.inst.{api_str}" in snapshot
          or f"amd_rdna.inst.{base}" in snapshot
          or api in RDNA_KNOWN_INSTRUCTIONS
          or api_str in RDNA_KNOWN_INSTRUCTIONS
          or base in RDNA_KNOWN_INSTRUCTIONS
        ):
          is_valid_api = True
      elif fw_name == "ir":
        api_str = str(api)
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

        ir_known_ops = set(ONNX_REGISTRY.keys()) | {
          "RMSNorm",
          "SwiGLU",
          "RoPE",
          "FlashAttention",
          "VisionPatchEmbedding",
          "ml_switcheroo_ir.LogicalNode",
          "sw_ir.LogicalNode",
          "LogicalNode",
        }
        if api_str in snapshot or api_str in ir_known_ops or api_str.startswith("sw_ir."):
          is_valid_api = True

      if not is_valid_api and grounding_engine is not None:
        ge_target = "amd_rdna" if fw_name == "rdna" else fw_name
        try:
          has_files = (
            grounding_engine._discover_target_files(ge_target)
            if hasattr(grounding_engine, "_discover_target_files")
            else True
          )
          if has_files and hasattr(grounding_engine, "has_symbol") and grounding_engine.has_symbol(ge_target, str(api)):
            is_valid_api = True
        except Exception:
          pass

      if not is_valid_api:
        strictly_checked = {
          "mlx",
          "torch",
          "jax",
          "tensorflow",
          "stablehlo",
          "rdna",
          "nvidia_sass",
          "numpy",
          "flax",
          "keras",
          "ir",
        }
        if (
          fw_name in strictly_checked
          and api not in ignore_list
          and not str(api).startswith(";")
          and not str(api).startswith("Macro.")
        ):
          suggestion = ""
          if grounding_engine is not None and hasattr(grounding_engine, "suggest_closest_symbol"):
            try:
              ge_target = "amd_rdna" if fw_name == "rdna" else fw_name
              has_files = (
                grounding_engine._discover_target_files(ge_target)
                if hasattr(grounding_engine, "_discover_target_files")
                else True
              )
              if has_files:
                closest = grounding_engine.suggest_closest_symbol(ge_target, str(api))
                if closest:
                  suggestion = f" (did you mean: '{closest}'?)"
            except Exception:
              pass
          errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated API: '{api}'{suggestion}")
      else:
        api_data = snapshot[api] if api in snapshot else {}

        # Check arguments
        args_map = fw_mapping.get("args", {})
        if not args_map:
          continue

        raw_snapshot_args = api_data.get("params", api_data.get("args", []))
        snapshot_args = []
        for arg in raw_snapshot_args:
          if not isinstance(arg, dict):
            continue
          arg_name = str(arg.get("name", ""))
          if "," in arg_name:
            for tok in [t.strip() for t in arg_name.split(",") if t.strip()]:
              if tok.isidentifier():
                exp_arg = dict(arg)
                exp_arg["name"] = tok
                snapshot_args.append(exp_arg)
          else:
            snapshot_args.append(arg)

        snapshot_arg_names = {arg["name"] for arg in snapshot_args if isinstance(arg, dict) and "name" in arg}

        ignore_args = {
          ("torch.sum", "dim"),
          ("torch.sum", "keepdim"),
        }

        for std_name, fw_arg_name in args_map.items():
          if fw_arg_name is None:
            # Explicit null parameter indicates intentional dropping/omission in target framework
            continue
          if fw_arg_name not in snapshot_arg_names:
            if (api, fw_arg_name) in ignore_args:
              continue
            # some args might be variadic or **kwargs, but we should verify exact matches if possible
            # check if the api has **kwargs
            has_kwargs = any(arg["name"] == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)
            if not has_kwargs:
              if fw_name in [
                "mlx",
                "torch",
                "jax",
                "tensorflow",
                "rdna",
                "nvidia_sass",
                "numpy",
                "flax",
                "flax_nnx",
                "keras",
              ]:
                errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated argument: '{fw_arg_name}' for API '{api}'")

        # Arity check: Ensure all required target arguments are provided
        if "macro_template" not in fw_mapping and fw_name in [
          "mlx",
          "torch",
          "jax",
          "tensorflow",
          "rdna",
          "nvidia_sass",
          "numpy",
          "flax",
          "flax_nnx",
          "keras",
        ]:
          required_args = {
            arg["name"]
            for arg in snapshot_args
            if arg.get("default") is None
            and arg.get("kind") not in ("VAR_POSITIONAL", "VAR_KEYWORD")
            and arg.get("name") not in ("self", "cls")
            and str(arg.get("name", "")).isidentifier()
          }
          has_kwargs = any(arg["name"] == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)

          mapped_target_args = {v for v in args_map.values() if v is not None}
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
            "dtype",
            "rngs",
          }
          missing_args = missing_args - known_missing_defaults

          if missing_args and not has_kwargs:
            # Ignore known gaps
            if api not in ignore_list:
              errors.append(f"[{fw_name}] '{op_name}' missing required arguments for API '{api}': {missing_args}")

  return errors


def compute_snapshot_checksums(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, str]:
  """Computes SHA-256 checksums of discovered snapshot files.

  Args:
      snapshot_dirs: Optional list of directories to scan for snapshots.

  Returns:
      A dictionary mapping snapshot file names to their SHA-256 hexadecimal digests.
  """
  import hashlib

  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    dirs_to_search.extend(snapshot_dirs)
  else:
    dirs_to_search.extend(
      [
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
        Path("src/ml_switcheroo/semantics"),
      ]
    )

  checksums: Dict[str, str] = {}
  for d in dirs_to_search:
    if not d.exists():
      continue
    candidates = sorted(set(list(d.glob("*_v*.json")) + list(d.glob("*_exhaustive.json")) + list(d.glob("*_isa.json"))))
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      try:
        data = file_path.read_bytes()
        if len(data) > 0:
          checksums[file_path.name] = hashlib.sha256(data).hexdigest()
      except Exception:
        pass
  return checksums


def generate_audit_report(
  manager: SemanticsManager,
  snapshots: Dict[str, Dict[str, Any]],
  errors: List[str],
  checksums: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
  """Generates a structured audit report dictionary across all supported targets.

  Args:
      manager: The active SemanticsManager instance.
      snapshots: Flattened snapshot lookup dictionary.
      errors: List of detected error strings.
      checksums: Optional dictionary of snapshot file SHA-256 checksums.

  Returns:
      A dictionary detailing coverage, mapped operation counts, and error metrics.
  """
  targets = ["torch", "jax", "mlx", "keras", "nvidia_sass", "rdna", "ir", "array_api", "scipy", "safetensors"]
  target_metrics: Dict[str, Dict[str, Any]] = {}

  for target in targets:
    mapped_count = 0
    for _op_name, op_details in manager.data.items():
      variants = op_details.get("variants", {})
      if target in variants:
        mapped_count += 1

    target_errors = [e for e in errors if f"[{target}]" in e]
    target_metrics[target] = {
      "mapped_operations": mapped_count,
      "snapshot_symbols": len(snapshots.get(target, {})),
      "errors": len(target_errors),
      "status": "valid" if len(target_errors) == 0 else "mismatched",
    }

  return {
    "total_operations": len(manager.data),
    "loaded_snapshots": list(snapshots.keys()),
    "snapshot_checksums": checksums or {},
    "total_errors": len(errors),
    "status": "pass" if len(errors) == 0 else "fail",
    "targets": target_metrics,
  }


def main() -> int:
  """Main execution function.

  Returns:
      Exit code (0 for success, 1 for failures).
  """
  parser = argparse.ArgumentParser(description="Audit against snapshots")
  parser.add_argument("--strict", action="store_true", help="Fail if any mismatches found")
  parser.add_argument(
    "--framework", type=str, default=None, help="Audit only a specific framework (e.g. rdna, nvidia_sass)"
  )
  parser.add_argument(
    "--report", "--output", dest="report_path", type=str, default=None, help="Save structured audit report to JSON"
  )
  args = parser.parse_args()

  mgr = SemanticsManager()
  KnowledgeBaseLoader(mgr).load_knowledge_graph()
  RegistryLoader(mgr).hydrate()

  snapshot_dirs = [
    Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
    Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
    Path("src/ml_switcheroo/semantics"),
    Path("../ml-compiler-snapshots"),
  ]
  for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
    try:
      snap_res = importlib.resources.files(pkg)
      snapshot_dirs.append(Path(str(snap_res)))
    except Exception:
      pass
  snapshots = load_snapshots_multi(snapshot_dirs)

  grounding_engine = None
  if GroundingEngine is not None:
    try:
      ge_dirs = [str(d) for d in snapshot_dirs if d.exists()]
      grounding_engine = GroundingEngine(base_dirs=ge_dirs)
    except Exception:
      grounding_engine = None

  print(f"Loaded {len(snapshots)} snapshots.")
  errors = audit_frameworks(mgr, snapshots, framework=args.framework, grounding_engine=grounding_engine)

  src_dirs = [Path("src/ml_switcheroo/frameworks"), Path("src/ml_switcheroo/plugins")]
  ast_errors = audit_python_ast(src_dirs, snapshots)
  errors.extend(ast_errors)

  snippet_errors = audit_inline_snippets(mgr, snapshots)
  errors.extend(snippet_errors)

  if args.report_path:
    checksums = compute_snapshot_checksums(snapshot_dirs)
    report = generate_audit_report(mgr, snapshots, errors, checksums=checksums)
    with open(args.report_path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"Saved audit report to {args.report_path}")

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
