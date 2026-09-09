"""ODL Variant Parity Expansion Tool.

Audits ODL operator definitions for missing target framework variants
(JAX, MLX, Keras, AMD RDNA, NVIDIA SASS) and expands variant coverage
using grounded snapshot APIs and standard ISA instructions/macros.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_against_snapshots import load_snapshots_multi  # noqa: E402
from scripts.drain_quarantine import find_snapshot_api_match, build_variant_entry  # noqa: E402

# Canonical mapping for basic ALU and Kernel Macros across RDNA and SASS
ISA_ALU_MACRO_MAP: Dict[str, Dict[str, str]] = {
  "Add": {"rdna": "v_add_f32", "nvidia_sass": "FADD"},
  "Sub": {"rdna": "v_sub_f32", "nvidia_sass": "FSUB"},
  "Subtract": {"rdna": "v_sub_f32", "nvidia_sass": "FSUB"},
  "Mul": {"rdna": "v_mul_f32", "nvidia_sass": "FMUL"},
  "Multiply": {"rdna": "v_mul_f32", "nvidia_sass": "FMUL"},
  "Fma": {"rdna": "v_fma_f32", "nvidia_sass": "FFMA"},
  "Min": {"rdna": "v_min_f32", "nvidia_sass": "FMNMX"},
  "Max": {"rdna": "v_max_f32", "nvidia_sass": "FMNMX"},
  "Abs": {"rdna": "v_abs_f32", "nvidia_sass": "FABS"},
  "Exp": {"rdna": "v_exp_f32", "nvidia_sass": "MUFU.EXP"},
  "Log": {"rdna": "v_log_f32", "nvidia_sass": "MUFU.LOG2"},
  "Sqrt": {"rdna": "v_sqrt_f32", "nvidia_sass": "MUFU.SQRT"},
  "Rsqrt": {"rdna": "v_rsq_f32", "nvidia_sass": "MUFU.RSQ"},
  "BitwiseAnd": {"rdna": "v_and_b32", "nvidia_sass": "LOP3_LUT"},
  "BitwiseOr": {"rdna": "v_or_b32", "nvidia_sass": "LOP3_LUT"},
  "BitwiseXor": {"rdna": "v_xor_b32", "nvidia_sass": "LOP3_LUT"},
  "Conv2d": {"rdna": "; Macro.Conv2d", "nvidia_sass": "Macro.Conv2d"},
  "Conv3d": {"rdna": "; Macro.Conv3d", "nvidia_sass": "Macro.Conv3d"},
  "Linear": {"rdna": "; Macro.Linear", "nvidia_sass": "Macro.Linear"},
  "Dense": {"rdna": "; Macro.Linear", "nvidia_sass": "Macro.Linear"},
  "ReLU": {"rdna": "; Macro.ReLU", "nvidia_sass": "Macro.ReLU"},
  "GELU": {"rdna": "; Macro.GELU", "nvidia_sass": "Macro.GELU"},
  "SiLU": {"rdna": "; Macro.SiLU", "nvidia_sass": "Macro.SiLU"},
  "BatchNorm": {"rdna": "; Macro.BatchNorm", "nvidia_sass": "Macro.BatchNorm"},
  "LayerNorm": {"rdna": "; Macro.LayerNorm", "nvidia_sass": "Macro.LayerNorm"},
  "RMSNorm": {"rdna": "; Macro.RMSNorm", "nvidia_sass": "Macro.RMSNorm"},
  "Softmax": {"rdna": "; Macro.Softmax", "nvidia_sass": "Macro.Softmax"},
  "Reshape": {"rdna": "; Macro.Reshape", "nvidia_sass": "Macro.Reshape"},
  "Flatten": {"rdna": "; Macro.Flatten", "nvidia_sass": "Macro.Flatten"},
}

CANONICAL_MATH_MAP: Dict[str, Dict[str, str]] = {
  "Abs": {"torch": "torch.abs", "jax": "jax.numpy.abs", "mlx": "mlx.core.abs", "keras": "keras.ops.abs"},
  "Add": {"torch": "torch.add", "jax": "jax.numpy.add", "mlx": "mlx.core.add", "keras": "keras.ops.add"},
  "Sub": {"torch": "torch.sub", "jax": "jax.numpy.subtract", "mlx": "mlx.core.subtract", "keras": "keras.ops.subtract"},
  "Subtract": {
    "torch": "torch.subtract",
    "jax": "jax.numpy.subtract",
    "mlx": "mlx.core.subtract",
    "keras": "keras.ops.subtract",
  },
  "Mul": {"torch": "torch.mul", "jax": "jax.numpy.multiply", "mlx": "mlx.core.multiply", "keras": "keras.ops.multiply"},
  "Multiply": {
    "torch": "torch.multiply",
    "jax": "jax.numpy.multiply",
    "mlx": "mlx.core.multiply",
    "keras": "keras.ops.multiply",
  },
  "Div": {"torch": "torch.div", "jax": "jax.numpy.divide", "mlx": "mlx.core.divide", "keras": "keras.ops.divide"},
  "Divide": {
    "torch": "torch.divide",
    "jax": "jax.numpy.divide",
    "mlx": "mlx.core.divide",
    "keras": "keras.ops.divide",
  },
  "Exp": {"torch": "torch.exp", "jax": "jax.numpy.exp", "mlx": "mlx.core.exp", "keras": "keras.ops.exp"},
  "Log": {"torch": "torch.log", "jax": "jax.numpy.log", "mlx": "mlx.core.log", "keras": "keras.ops.log"},
  "Sin": {"torch": "torch.sin", "jax": "jax.numpy.sin", "mlx": "mlx.core.sin", "keras": "keras.ops.sin"},
  "Cos": {"torch": "torch.cos", "jax": "jax.numpy.cos", "mlx": "mlx.core.cos", "keras": "keras.ops.cos"},
  "Sqrt": {"torch": "torch.sqrt", "jax": "jax.numpy.sqrt", "mlx": "mlx.core.sqrt", "keras": "keras.ops.sqrt"},
  "Rsqrt": {"torch": "torch.rsqrt", "jax": "jax.lax.rsqrt", "mlx": "mlx.core.rsqrt", "keras": "keras.ops.rsqrt"},
  "Tanh": {"torch": "torch.tanh", "jax": "jax.numpy.tanh", "mlx": "mlx.core.tanh", "keras": "keras.ops.tanh"},
  "ReLU": {
    "torch": "torch.nn.functional.relu",
    "jax": "jax.nn.relu",
    "mlx": "mlx.nn.relu",
    "keras": "keras.activations.relu",
  },
  "GELU": {
    "torch": "torch.nn.functional.gelu",
    "jax": "jax.nn.gelu",
    "mlx": "mlx.nn.gelu",
    "keras": "keras.activations.gelu",
  },
  "SiLU": {
    "torch": "torch.nn.functional.silu",
    "jax": "jax.nn.silu",
    "mlx": "mlx.nn.silu",
    "keras": "keras.activations.silu",
  },
  "Sigmoid": {
    "torch": "torch.sigmoid",
    "jax": "jax.nn.sigmoid",
    "mlx": "mlx.core.sigmoid",
    "keras": "keras.activations.sigmoid",
  },
  "MatMul": {
    "torch": "torch.matmul",
    "jax": "jax.numpy.matmul",
    "mlx": "mlx.core.matmul",
    "keras": "keras.ops.matmul",
  },
  "Linear": {
    "torch": "torch.nn.Linear",
    "flax_nnx": "flax.nnx.Linear",
    "mlx": "mlx.nn.Linear",
    "keras": "keras.layers.Dense",
  },
  "Dense": {
    "torch": "torch.nn.Linear",
    "flax_nnx": "flax.nnx.Linear",
    "mlx": "mlx.nn.Linear",
    "keras": "keras.layers.Dense",
  },
  "LayerNorm": {
    "torch": "torch.nn.LayerNorm",
    "flax_nnx": "flax.nnx.LayerNorm",
    "mlx": "mlx.nn.LayerNorm",
    "keras": "keras.layers.LayerNormalization",
  },
  "RMSNorm": {
    "torch": "torch.nn.RMSNorm",
    "flax_nnx": "flax.nnx.RMSNorm",
    "mlx": "mlx.nn.RMSNorm",
    "keras": "keras.layers.RMSNormalization",
  },
  "Tan": {"torch": "torch.tan", "jax": "jax.numpy.tan", "mlx": "mlx.core.tan", "keras": "keras.ops.tan"},
  "Floor": {"torch": "torch.floor", "jax": "jax.numpy.floor", "mlx": "mlx.core.floor", "keras": "keras.ops.floor"},
  "Ceil": {"torch": "torch.ceil", "jax": "jax.numpy.ceil", "mlx": "mlx.core.ceil", "keras": "keras.ops.ceil"},
  "Round": {"torch": "torch.round", "jax": "jax.numpy.round", "mlx": "mlx.core.round", "keras": "keras.ops.round"},
  "Sign": {"torch": "torch.sign", "jax": "jax.numpy.sign", "mlx": "mlx.core.sign", "keras": "keras.ops.sign"},
  "Negative": {
    "torch": "torch.negative",
    "jax": "jax.numpy.negative",
    "mlx": "mlx.core.negative",
    "keras": "keras.ops.negative",
  },
  "Square": {"torch": "torch.square", "jax": "jax.numpy.square", "mlx": "mlx.core.square", "keras": "keras.ops.square"},
  "Log2": {"torch": "torch.log2", "jax": "jax.numpy.log2", "mlx": "mlx.core.log2", "keras": "keras.ops.log2"},
  "Log10": {"torch": "torch.log10", "jax": "jax.numpy.log10", "mlx": "mlx.core.log10", "keras": "keras.ops.log10"},
  "Maximum": {
    "torch": "torch.maximum",
    "jax": "jax.numpy.maximum",
    "mlx": "mlx.core.maximum",
    "keras": "keras.ops.maximum",
  },
  "Minimum": {
    "torch": "torch.minimum",
    "jax": "jax.numpy.minimum",
    "mlx": "mlx.core.minimum",
    "keras": "keras.ops.minimum",
  },
  "Sum": {"torch": "torch.sum", "jax": "jax.numpy.sum", "mlx": "mlx.core.sum", "keras": "keras.ops.sum"},
  "Mean": {"torch": "torch.mean", "jax": "jax.numpy.mean", "mlx": "mlx.core.mean", "keras": "keras.ops.mean"},
  "Softmax": {
    "torch": "torch.nn.functional.softmax",
    "jax": "jax.nn.softmax",
    "mlx": "mlx.core.softmax",
    "keras": "keras.activations.softmax",
  },
  "Dropout": {
    "torch": "torch.nn.Dropout",
    "flax_nnx": "flax.nnx.Dropout",
    "mlx": "mlx.nn.Dropout",
    "keras": "keras.layers.Dropout",
  },
  "Conv1d": {
    "torch": "torch.nn.Conv1d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv1d",
    "keras": "keras.layers.Conv1D",
  },
  "Conv2d": {
    "torch": "torch.nn.Conv2d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv2d",
    "keras": "keras.layers.Conv2D",
  },
  "Conv3d": {
    "torch": "torch.nn.Conv3d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv3d",
    "keras": "keras.layers.Conv3D",
  },
}


def expand_op_variants(
  odl_data: Dict[str, Any],
  snapshots: Dict[str, Dict[str, Any]],
) -> int:
  """Expands missing target framework variants in an ODL dictionary.

  Args:
      odl_data: The parsed ODL YAML dictionary.
      snapshots: Loaded ground-truth snapshot dictionary.

  Returns:
      Number of newly added variants.
  """
  variants = odl_data.setdefault("variants", {})
  op_name = odl_data.get("operation", "")

  added_count = 0

  # Candidate search names: op_name and function name from torch variant
  search_names: List[str] = [op_name]
  torch_api = variants.get("torch", {}).get("api")
  if torch_api:
    fn_name = torch_api.split(".")[-1]
    if fn_name not in search_names:
      search_names.append(fn_name)

  # 1. Expand JAX, MLX, Keras from snapshots
  for fw in ("jax", "mlx", "keras"):
    if fw not in variants and fw in snapshots:
      fw_snap = snapshots[fw]
      match: Optional[Tuple[str, Dict[str, Any]]] = None
      for candidate in search_names:
        match = find_snapshot_api_match(candidate, fw_snap)
        if match:
          break

      # Fallback for JAX from paxml or flax_nnx jnp references
      if not match and fw == "jax":
        for source_fw in ("paxml", "flax_nnx"):
          src_api = variants.get(source_fw, {}).get("api", "")
          if src_api.startswith("jnp."):
            jnp_fn = src_api.split(".")[-1]
            match = find_snapshot_api_match(jnp_fn, fw_snap)
            if match:
              break

      if match:
        api_path, api_data = match
        variants[fw] = build_variant_entry(api_path, api_data)
        added_count += 1

  # 2. Expand RDNA and NVIDIA SASS from ISA_ALU_MACRO_MAP (case-insensitive, strip trailing _)
  op_clean = op_name.rstrip("_").lower()
  isa_entry = None
  for k, v in ISA_ALU_MACRO_MAP.items():
    if k.lower() == op_clean:
      isa_entry = v
      break

  if isa_entry:
    if "rdna" not in variants and "rdna" in isa_entry:
      variants["rdna"] = {"api": isa_entry["rdna"], "args": {}}
      added_count += 1
    if "nvidia_sass" not in variants and "nvidia_sass" in isa_entry:
      variants["nvidia_sass"] = {"api": isa_entry["nvidia_sass"], "args": {}}
      added_count += 1

  # 3. Expand missing high-level frameworks from CANONICAL_MATH_MAP
  math_entry = None
  for k, v in CANONICAL_MATH_MAP.items():
    if k.lower() == op_clean:
      math_entry = v
      break

  if math_entry:
    for fw, api_call in math_entry.items():
      if fw not in variants:
        variants[fw] = {"api": api_call, "args": {}}
        added_count += 1

  return added_count


def expand_parity_across_odl(
  odl_dir: Path,
  snapshots: Dict[str, Dict[str, Any]],
  dry_run: bool = False,
) -> Tuple[int, int]:
  """Iterates through ODL definitions and expands missing variants.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      snapshots: Ground-truth snapshot dictionaries.
      dry_run: If True, calculates changes without writing to disk.

  Returns:
      A tuple of (modified_file_count, total_variants_added).
  """
  modified_files = 0
  total_added = 0

  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except Exception:
      continue

    if not isinstance(data, dict):
      continue

    added = expand_op_variants(data, snapshots)
    if added > 0:
      modified_files += 1
      total_added += added
      if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
          yaml.dump(data, f, sort_keys=False, indent=2)

  return modified_files, total_added


def main(args: Optional[Sequence[str]] = None) -> int:
  """Command-line interface for ODL variant parity expansion.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Expand ODL variant parity across frameworks")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Compute changes without writing to disk",
  )
  parsed = parser.parse_args(args)

  snapshot_dirs = [
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
    Path("../ml-compiler-snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)

  mod_files, added_variants = expand_parity_across_odl(
    odl_dir=parsed.odl_dir,
    snapshots=snapshots,
    dry_run=parsed.dry_run,
  )

  print(f"Expanded parity across {mod_files} ODL files (+{added_variants} variants).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
