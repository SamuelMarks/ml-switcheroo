"""AST and Snippet static analysis tools for snapshot grounding."""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ml_switcheroo.semantics.manager import SemanticsManager
from scripts.snapshots.loader import GroundingEngine

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


def extract_api_calls(file_path: Path) -> Set[str]:
  """Extracts fully qualified API calls from a Python file.

  Args:
      file_path: Path to the Python source file to inspect.

  Returns:
      Set of fully qualified API call strings discovered in the file.
  """
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
  """Audits programmatic API calls inside declarative inline snippets (e.g. macro_template).

  Args:
      manager: The hydrated SemanticsManager containing operation variants.
      snapshots: Mapping of framework keys to flattened snapshot symbol dictionaries.

  Returns:
      List of error messages for hallucinated API calls inside inline snippets.
  """
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
  """Audits programmatic API calls in Python files against snapshots.

  Args:
      src_dirs: List of source directories to recursively scan for Python files.
      snapshots: Mapping of framework keys to flattened snapshot symbol dictionaries.

  Returns:
      List of error messages for ungrounded programmatic API calls.
  """
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

        try:
          from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

          canonical_onnx = {op["name"] for op in _load_onnx_ops() if isinstance(op, dict) and "name" in op}
        except Exception:
          canonical_onnx = set()

        ir_known_ops = (
          set(ONNX_REGISTRY.keys())
          | canonical_onnx
          | {
            "Add",
            "BatchNormalization",
            "Conv",
            "LayerNormalization",
            "Gemm",
            "MatMul",
            "Relu",
            "Softmax",
            "RMSNorm",
            "SwiGLU",
            "RoPE",
            "FlashAttention",
            "VisionPatchEmbedding",
            "ml_switcheroo_ir.LogicalNode",
            "sw_ir.LogicalNode",
            "LogicalNode",
          }
        )
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
