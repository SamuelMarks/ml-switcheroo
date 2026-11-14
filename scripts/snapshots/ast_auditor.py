"""AST and Snippet static analysis tools for snapshot grounding."""

import ast
import logging
import typing
from pathlib import Path

from pydantic import JsonValue

from ml_switcheroo.enums import RuntimeTensorOrValue
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
    elif isinstance(node, ast.ImportFrom) and node.module:
      for name in node.names:
        aliases[name.asname or name.name] = f"{node.module}.{name.name}"

  apis: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
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
    for fw_mapping in variants.values():
      if isinstance(fw_mapping, dict) and "macro_template" in fw_mapping:
        snippet = fw_mapping["macro_template"]
        if not isinstance(snippet, str):
          continue
        try:
          tree = ast.parse(snippet)
        except (SyntaxError, ValueError, TypeError) as exc:
          logging.getLogger(__name__).debug("Skipping unparseable example snippet: %s", exc)
          continue

        for node in ast.walk(tree):
          if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
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
        if call.endswith((".model_validate", ".model_dump", ".from_json", ".to_json")):
          continue

        root_module = call.split(".")[0]
        if root_module in framework_prefixes:
          fw_key = framework_prefixes[root_module]
          if fw_key in snapshots:
            snapshot = snapshots[fw_key]
            if isinstance(snapshot, typing.Mapping) and call not in snapshot:
              errors.append(f"[{fw_key}] Programmatic API call hallucinated in {file_path}: '{call}'")

  return errors


_STRICTLY_CHECKED_FRAMEWORKS: frozenset[str] = frozenset(
  {
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
)

_SIGNATURE_CHECKED_FRAMEWORKS: frozenset[str] = frozenset(
  {
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
  }
)

_IR_CORE_OPS: frozenset[str] = frozenset(
  {
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


def _init_grounding_engine(grounding_engine: GroundingEngineProtocol | None) -> GroundingEngineProtocol | None:
  """Instantiate a default ``GroundingEngine`` if none was supplied and one is available.

  Args:
      grounding_engine: Optional caller-provided ``GroundingEngineProtocol`` instance.

  Returns:
      Resolved ``GroundingEngineProtocol`` instance or ``None``.
  """
  if grounding_engine is None and GroundingEngine is not None:
    try:
      return GroundingEngine()
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
      logging.getLogger(__name__).debug("GroundingEngine unavailable: %s", exc)
      return None
  return grounding_engine


def _check_ir_api_valid(api: str, snapshot: typing.Mapping[str, RuntimeTensorOrValue]) -> bool:
  """Verify whether ``api`` is a valid canonical or snapshot-backed IR operation.

  Args:
      api: Candidate IR operation identifier.
      snapshot: Flattened IR snapshot dictionary.

  Returns:
      True if ``api`` is recognized in IR schemas or snapshots.
  """
  from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

  try:
    from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

    canonical_onnx = {str(op["name"]) for op in _load_onnx_ops() if isinstance(op, dict) and "name" in op}
  except (ImportError, OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
    logging.getLogger(__name__).debug("ONNX canonical spec lookup skipped: %s", exc)
    canonical_onnx = set()

  ir_known_ops = set(ONNX_REGISTRY.keys()) | canonical_onnx | _IR_CORE_OPS
  return api in snapshot or api in ir_known_ops or api.startswith("sw_ir.")


def _check_isa_api_valid(fw_name: str, api: str, snapshot: typing.Mapping[str, RuntimeTensorOrValue]) -> bool:
  """Verify whether ``api`` is a valid ``nvidia_sass`` or ``rdna`` instruction mnemonic.

  Args:
      fw_name: Dialect identifier (``"nvidia_sass"`` or ``"rdna"``).
      api: Candidate instruction mnemonic.
      snapshot: Flattened ISA snapshot dictionary.

  Returns:
      True if ``api`` matches a known or snapshot-grounded instruction.
  """
  if fw_name == "nvidia_sass":
    api_str = api.upper()
    base = api_str.split(".")[0].split("_")[0]
    candidates = {api_str, base, base.lstrip("U"), f"nvidia_sass.inst.{api_str}", f"nvidia_sass.inst.{base}"}
    return bool(candidates & set(snapshot)) or bool({api, api_str, base} & set(NVIDIA_SASS_KNOWN_INSTRUCTIONS))

  api_str = api.lower()
  base = api_str.split(".")[0]
  candidates = {api_str, base, f"amd_rdna.inst.{api_str}", f"amd_rdna.inst.{base}"}
  return bool(candidates & set(snapshot)) or bool({api, api_str, base} & set(RDNA_KNOWN_INSTRUCTIONS))


def _resolve_api_against_snapshots(
  fw_name: str,
  api: str,
  snapshot: typing.Mapping[str, RuntimeTensorOrValue],
  snapshots: typing.Mapping[str, typing.Mapping[str, RuntimeTensorOrValue] | RuntimeTensorOrValue],
) -> tuple[bool, typing.Mapping[str, RuntimeTensorOrValue]]:
  """Check whether ``api`` exists in ``snapshot`` or delegated sub-framework snapshots.

  Args:
      fw_name: Framework identifier being audited.
      api: Candidate dotted API string.
      snapshot: Primary snapshot dictionary for ``fw_name``.
      snapshots: Full multi-framework snapshot map.

  Returns:
      Tuple ``(is_valid, resolved_snapshot)``.
  """
  if api in snapshot:
    return True, snapshot
  if fw_name in ("jax", "flax_nnx") and api.startswith("optax."):
    optax_snap = snapshots.get("optax")
    if isinstance(optax_snap, typing.Mapping) and api in optax_snap:
      return True, optax_snap
  elif fw_name == "flax_nnx" and (api.startswith(("jax.numpy.", "jnp."))):
    jax_snap = snapshots.get("jax")
    if isinstance(jax_snap, typing.Mapping) and api in jax_snap:
      return True, jax_snap
  elif fw_name in ("nvidia_sass", "rdna"):
    return _check_isa_api_valid(fw_name, api, snapshot), snapshot
  elif fw_name == "ir":
    return _check_ir_api_valid(api, snapshot), snapshot
  return False, snapshot


def _grounding_has_target_files(grounding_engine: GroundingEngineProtocol, ge_target: str) -> bool:
  """Check whether ``grounding_engine`` discovers target files for ``ge_target``.

  Args:
      grounding_engine: Active ``GroundingEngineProtocol`` instance.
      ge_target: Normalized target name for grounding lookup.

  Returns:
      True if target snapshot files exist or ``_discover_target_files`` is unimplemented.
  """
  try:
    return bool(grounding_engine._discover_target_files(ge_target))
  except AttributeError:
    return True


def _check_grounding_symbol(
  grounding_engine: GroundingEngineProtocol | None,
  fw_name: str,
  api: str,
) -> bool:
  """Query ``grounding_engine.has_symbol`` safely for ``(fw_name, api)``.

  Args:
      grounding_engine: Optional grounding engine instance.
      fw_name: Framework identifier.
      api: Candidate API path.

  Returns:
      True if ``grounding_engine`` confirms ``api`` exists.
  """
  if grounding_engine is None:
    return False
  ge_target = "amd_rdna" if fw_name == "rdna" else fw_name
  try:
    return _grounding_has_target_files(grounding_engine, ge_target) and grounding_engine.has_symbol(ge_target, api)
  except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
    logging.getLogger(__name__).debug("GroundingEngine.has_symbol failed for %s:%s: %s", ge_target, api, exc)
    return False


def _format_hallucinated_api_error(
  fw_name: str,
  op_name: str,
  api: str,
  grounding_engine: GroundingEngineProtocol | None,
) -> str | None:
  """Format a hallucinated API diagnostic string (with optional closest-symbol suggestion).

  Args:
      fw_name: Framework identifier.
      op_name: Abstract operation name.
      api: Unverified target API string.
      grounding_engine: Optional grounding engine for closest-symbol suggestions.

  Returns:
      Formatted error message string if ``fw_name`` is strictly checked, else ``None``.
  """
  if fw_name not in _STRICTLY_CHECKED_FRAMEWORKS or api.startswith((";", "Macro.")):
    return None
  suggestion = ""
  if grounding_engine is not None:
    ge_target = "amd_rdna" if fw_name == "rdna" else fw_name
    try:
      if _grounding_has_target_files(grounding_engine, ge_target):
        closest = grounding_engine.suggest_closest_symbol(ge_target, api)
        if closest:
          suggestion = f" (did you mean: '{closest}'?)"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
      logging.getLogger(__name__).debug(
        "GroundingEngine.suggest_closest_symbol failed for %s:%s: %s", ge_target, api, exc
      )
  return f"[{fw_name}] '{op_name}' maps to hallucinated API: '{api}'{suggestion}"


def _expand_snapshot_args(api_data: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
  """Normalize and split comma-separated composite parameter names from ``api_data``.

  Args:
      api_data: Snapshot entry dictionary for a single API symbol.

  Returns:
      Expanded list of parameter metadata dictionaries.
  """
  raw_snapshot_args = api_data.get("params", api_data.get("args", []))
  snapshot_args: list[dict[str, JsonValue]] = []
  if not isinstance(raw_snapshot_args, list):
    return snapshot_args
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
  return snapshot_args


def _compute_missing_required_args(
  api_data: dict[str, JsonValue],
  snapshot_args: list[dict[str, JsonValue]],
  mapped_target_args: set[str],
) -> set[str] | None:
  """Compute the minimal set of unmapped required target parameters across overloads.

  Args:
      api_data: Snapshot entry dictionary for the target API.
      snapshot_args: Expanded primary parameter list.
      mapped_target_args: Set of target parameter names covered by the ODL mapping.

  Returns:
      Minimal set of missing required parameter names across overloads.
  """
  overloads = api_data.get("overloads", [])
  candidate_param_lists: list[list[dict[str, JsonValue]]] = [snapshot_args]
  if isinstance(overloads, list):
    for ov in overloads:
      if isinstance(ov, dict):
        ov_params = ov.get("params")
        if isinstance(ov_params, list):
          candidate_param_lists.append([p for p in ov_params if isinstance(p, dict)])

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
    if api_data.get("api_path") and len(mapped_target_args & req_args) < len(req_args):
      req_args = set()
    missing = req_args - mapped_target_args
    if best_missing is None or len(missing) < len(best_missing):
      best_missing = missing
  return best_missing


def _audit_signature_parity(
  fw_name: str,
  op_name: str,
  api: str,
  fw_mapping: dict[str, JsonValue],
  std_arg_names: set[str],
  api_data: dict[str, JsonValue],
) -> list[str]:
  """Verify mapped parameter names and required arity against grounded snapshot signatures.

  Args:
      fw_name: Framework identifier.
      op_name: Abstract operation name.
      api: Grounded target API path.
      fw_mapping: ODL variant dictionary for ``fw_name``.
      std_arg_names: Set of canonical ``std_args`` names on the operation.
      api_data: Snapshot metadata dictionary for ``api``.

  Returns:
      List of signature/arity mismatch diagnostic strings.
  """
  raw_args_map = fw_mapping.get("args")
  if not isinstance(raw_args_map, dict) or not raw_args_map or fw_name not in _SIGNATURE_CHECKED_FRAMEWORKS:
    return []

  snapshot_args = _expand_snapshot_args(api_data)
  snapshot_arg_names = {str(arg["name"]) for arg in snapshot_args if "name" in arg}
  has_kwargs = any(arg.get("name") == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)

  errors: list[str] = [
    f"[{fw_name}] '{op_name}' maps to hallucinated argument: '{fw_arg_name}' for API '{api}'"
    for fw_arg_name in raw_args_map.values()
    if fw_arg_name is not None and str(fw_arg_name) not in snapshot_arg_names and not has_kwargs
  ]

  if "macro_template" not in fw_mapping and not has_kwargs:
    raw_inject = fw_mapping.get("inject_args")
    inject_args = set(raw_inject.keys()) if isinstance(raw_inject, dict) else set()
    passthrough_std = std_arg_names - set(raw_args_map.keys())
    mapped_target_args = {str(v) for v in raw_args_map.values() if v is not None} | passthrough_std | inject_args
    best_missing = _compute_missing_required_args(api_data, snapshot_args, mapped_target_args)
    if best_missing:
      errors.append(f"[{fw_name}] '{op_name}' missing required arguments for API '{api}': {best_missing}")

  return errors


def _extract_std_arg_names(op_details: dict[str, JsonValue]) -> set[str]:
  """Extract canonical ``std_args`` parameter names from ``op_details``.

  Args:
      op_details: Operation specification dictionary from ``SemanticsManager.data``.

  Returns:
      Set of canonical parameter name strings.
  """
  raw_std_args = op_details.get("std_args")
  std_args = raw_std_args if isinstance(raw_std_args, list) else []
  return {
    str(s.get("name")) if isinstance(s, dict) else (str(s[0]) if isinstance(s, (list, tuple)) and s else str(s))
    for s in std_args
  }


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
  active_engine = _init_grounding_engine(grounding_engine)

  for op_name, op_details in manager.data.items():
    raw_variants = op_details.get("variants")
    variants = raw_variants if isinstance(raw_variants, dict) else {}
    std_arg_names = _extract_std_arg_names(op_details)

    for fw_name, fw_mapping in variants.items():
      if not isinstance(fw_mapping, dict) or (framework is not None and fw_name != framework) or fw_name not in snapshots:
        continue
      raw_snapshot = snapshots[fw_name]
      raw_api = fw_mapping.get("api")
      if not isinstance(raw_snapshot, typing.Mapping) or not isinstance(raw_api, str) or not raw_api:
        continue

      is_valid_api, resolved_snap = _resolve_api_against_snapshots(fw_name, raw_api, raw_snapshot, snapshots)
      if not is_valid_api and _check_grounding_symbol(active_engine, fw_name, raw_api):
        is_valid_api = True

      if not is_valid_api:
        err = _format_hallucinated_api_error(fw_name, op_name, raw_api, active_engine)
        if err is not None:
          errors.append(err)
        continue

      raw_api_data = resolved_snap.get(raw_api, {})
      api_data = raw_api_data if isinstance(raw_api_data, dict) else {}
      errors.extend(_audit_signature_parity(fw_name, op_name, raw_api, fw_mapping, std_arg_names, api_data))

  return errors
