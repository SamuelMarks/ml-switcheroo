"""AST and Snippet static analysis tools for snapshot grounding."""

from ml_switcheroo.enums import RuntimeTensorOrValue
import ast
import typing
from pathlib import Path

from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager
from scripts.snapshots.loader import GroundingEngine, GroundingEngineProtocol

RDNA_KNOWN_INSTRUCTIONS: set[str] = {
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

NVIDIA_SASS_KNOWN_INSTRUCTIONS: set[str] = {
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


def extract_api_calls(file_path: Path) -> set[str]:
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

  aliases: dict[str, str] = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for name in node.names:
        aliases[name.asname or name.name] = name.name
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        for name in node.names:
          aliases[name.asname or name.name] = f"{node.module}.{name.name}"

  apis: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      if isinstance(node.func, ast.Attribute):
        chain: list[str] = []
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


def audit_inline_snippets(
  manager: SemanticsManager,
  snapshots: typing.Mapping[
    str,
    typing.Mapping[str, RuntimeTensorOrValue] | RuntimeTensorOrValue,
  ],
) -> list[str]:
  """Audits programmatic API calls inside declarative inline snippets (e.g. macro_template).

  Args:
      manager: The hydrated SemanticsManager containing operation variants.
      snapshots: Mapping of framework keys to flattened snapshot symbol dictionaries.

  Returns:
      List of error messages for hallucinated API calls inside inline snippets.
  """
  errors: list[str] = []

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

  for op_name, op_details in manager.data.items():
    variants = op_details.get("variants")
    if not isinstance(variants, dict):
      continue
    for fw_name, fw_mapping in variants.items():
      if isinstance(fw_mapping, dict) and "macro_template" in fw_mapping:
        snippet = fw_mapping["macro_template"]
        if not isinstance(snippet, str):
          continue
        try:
          tree = ast.parse(snippet)
        except Exception:
          continue

        for node in ast.walk(tree):
          if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
              chain: list[str] = []
              curr: ast.expr = node.func
              while isinstance(curr, ast.Attribute):
                chain.append(curr.attr)
                curr = curr.value
              if isinstance(curr, ast.Name):
                chain.append(curr.id)
                chain.reverse()
                root = chain[0]
                full_api = root + "." + ".".join(chain[1:])

                if root in framework_prefixes:
                  fw_key = framework_prefixes[root]
                  if fw_key in snapshots:
                    snapshot = snapshots[fw_key]
                    if isinstance(snapshot, typing.Mapping) and full_api not in snapshot:
                      errors.append(f"[{fw_key}] Inline snippet hallucinated API in {op_name}: '{full_api}'")

  return errors


def audit_python_ast(
  src_dirs: list[Path],
  snapshots: typing.Mapping[
    str,
    typing.Mapping[str, RuntimeTensorOrValue] | RuntimeTensorOrValue,
  ],
) -> list[str]:
  """Audits programmatic API calls in Python files against snapshots.

  Args:
      src_dirs: List of source directories to recursively scan for Python files.
      snapshots: Mapping of framework keys to flattened snapshot symbol dictionaries.

  Returns:
      List of error messages for ungrounded programmatic API calls.
  """
  errors: list[str] = []

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

  for src_dir in src_dirs:
    if not src_dir.exists():
      continue
    for file_path in src_dir.rglob("*.py"):
      calls = extract_api_calls(file_path)
      for call in calls:
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
            if isinstance(snapshot, typing.Mapping) and call not in snapshot:
              errors.append(f"[{fw_key}] Programmatic API call hallucinated in {file_path}: '{call}'")

  return errors


def audit_frameworks(
  manager: SemanticsManager,
  snapshots: typing.Mapping[
    str,
    typing.Mapping[str, RuntimeTensorOrValue] | RuntimeTensorOrValue,
  ],
  framework: str | None = None,
  grounding_engine: GroundingEngineProtocol | None = None,
) -> list[str]:
  """Audits the known manager data against the snapshots.

  Args:
      manager: The hydrated SemanticsManager.
      snapshots: The loaded snapshots.
      framework: Optional framework identifier to filter auditing.
      grounding_engine: Optional GroundingEngine instance for anti-hallucination verification.

  Returns:
      A list of error strings.
  """
  errors: list[str] = []

  if grounding_engine is None and GroundingEngine is not None:
    try:
      grounding_engine = GroundingEngine()
    except Exception:
      grounding_engine = None

  for op_name, op_details in manager.data.items():
    raw_variants = op_details.get("variants")
    variants = raw_variants if isinstance(raw_variants, dict) else {}
    raw_std_args = op_details.get("std_args")
    std_args = raw_std_args if isinstance(raw_std_args, list) else []
    std_arg_names = {
      str(s.get("name")) if isinstance(s, dict) else (str(s[0]) if isinstance(s, (list, tuple)) and s else str(s))
      for s in std_args
    }
    for fw_name, fw_mapping in variants.items():
      if not isinstance(fw_mapping, dict):
        continue
      if framework is not None and fw_name != framework:
        continue
      if fw_name not in snapshots:
        # We might not have a snapshot for every framework in the matrix
        continue

      raw_snapshot = snapshots[fw_name]
      if not isinstance(raw_snapshot, typing.Mapping):
        continue
      snapshot: typing.Mapping[str, RuntimeTensorOrValue] = raw_snapshot
      raw_api = fw_mapping.get("api")
      if not isinstance(raw_api, str) or not raw_api:
        continue
      api = raw_api

      is_valid_api = False
      if api in snapshot:
        is_valid_api = True
      elif fw_name in ("jax", "flax_nnx") and api.startswith("optax."):
        optax_snap = snapshots.get("optax")
        if isinstance(optax_snap, typing.Mapping) and api in optax_snap:
          is_valid_api = True
          snapshot = optax_snap
      elif fw_name == "flax_nnx" and (api.startswith("jax.numpy.") or api.startswith("jnp.")):
        jax_snap = snapshots.get("jax")
        if isinstance(jax_snap, typing.Mapping) and api in jax_snap:
          is_valid_api = True
          snapshot = jax_snap
      elif fw_name == "nvidia_sass":
        api_str = api.upper()
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
        api_str = api.lower()
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
        api_str = api
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

        try:
          from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

          canonical_onnx = {str(op["name"]) for op in _load_onnx_ops() if isinstance(op, dict) and "name" in op}
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
          if has_files and hasattr(grounding_engine, "has_symbol") and grounding_engine.has_symbol(ge_target, api):
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
        if fw_name in strictly_checked and not api.startswith(";") and not api.startswith("Macro."):
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
                closest = grounding_engine.suggest_closest_symbol(ge_target, api)
                if closest:
                  suggestion = f" (did you mean: '{closest}'?)"
            except Exception:
              pass
          errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated API: '{api}'{suggestion}")
      else:
        raw_api_data = snapshot[api] if api in snapshot else {}
        api_data = raw_api_data if isinstance(raw_api_data, dict) else {}

        # Check arguments
        raw_args_map = fw_mapping.get("args")
        if not isinstance(raw_args_map, dict) or not raw_args_map:
          continue
        args_map = raw_args_map

        raw_snapshot_args = api_data.get("params", api_data.get("args", []))
        snapshot_args: list[dict[str, JsonValue]] = []
        if isinstance(raw_snapshot_args, list):
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

        snapshot_arg_names = {str(arg["name"]) for arg in snapshot_args if "name" in arg}

        for std_name, fw_arg_name in args_map.items():
          if fw_arg_name is None:
            # Explicit null parameter indicates intentional dropping/omission in target framework
            continue
          if str(fw_arg_name) not in snapshot_arg_names:
            # check if the api has **kwargs
            has_kwargs = any(arg.get("name") == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)
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

        # Arity check: Ensure all required target arguments are provided when std_args covers the full signature
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
          overloads = api_data.get("overloads", [])
          candidate_param_lists: list[list[dict[str, JsonValue]]] = [snapshot_args]
          if isinstance(overloads, list):
            for ov in overloads:
              if isinstance(ov, dict):
                ov_params = ov.get("params")
                if isinstance(ov_params, list):
                  candidate_param_lists.append([p for p in ov_params if isinstance(p, dict)])

          has_kwargs = any(arg.get("name") == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)
          raw_inject = fw_mapping.get("inject_args")
          inject_args = set(raw_inject.keys()) if isinstance(raw_inject, dict) else set()
          passthrough_std = std_arg_names - set(args_map.keys())
          mapped_target_args = {str(v) for v in args_map.values() if v is not None} | passthrough_std | inject_args

          best_missing: set[str] | None = None
          for param_list in candidate_param_lists:
            req_args = {
              str(arg["name"])
              for arg in param_list
              if arg.get("default") is None
              and arg.get("kind") not in ("VAR_POSITIONAL", "VAR_KEYWORD")
              and arg.get("name") not in ("self", "cls")
              and str(arg.get("name", "")).isidentifier()
            }
            # Standardized parameter aliases (e.g. input -> x, dim -> axis) or positional call-site arguments
            # when the ODL operation only remaps keyword options (len(std_args) < len(req_args) or api_data.get("api_path"))
            if api_data.get("api_path") and len(mapped_target_args & req_args) < len(req_args):
              req_args = set()
            missing = req_args - mapped_target_args
            if best_missing is None or len(missing) < len(best_missing):
              best_missing = missing

          if best_missing and not has_kwargs:
            errors.append(f"[{fw_name}] '{op_name}' missing required arguments for API '{api}': {best_missing}")

  return errors
