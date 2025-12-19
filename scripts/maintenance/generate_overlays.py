#!/usr/bin/env python3
"""
Unified Static Knowledge Base Generator.

This script consolidates the logic previously scattered across `src/ml_switcheroo/snapshots/*.py`.
It populates the Semantic Knowledge Base (Hub and Spokes) with manually curated mapping rules
that cover complex edge cases not easily handled by automated discovery.

Generators included:
- BatchNorm (State unwrapping)
- Casting/Dtypes (String aliases)
- Checkpointing (IO/Key Mapping)
- Clamping (Argument renaming)
- Flatten/Reshape/View variants
- Functional Transforms (vmap, grad)
- Gather/Scatter/Padding
- Optimizers/Schedulers
- Vision ops (OneHot, TopK)
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Locate the project root relative to this script
# scripts/maintenance/generate_overlays.py -> ../../ -> root
ROOT_DIR = Path(__file__).resolve().parents[2]
SEMANTICS_DIR = ROOT_DIR / "src/ml_switcheroo/semantics"
SNAPSHOTS_DIR = ROOT_DIR / "src/ml_switcheroo/snapshots"


def _read_json(path: Path) -> Dict[str, Any]:
  if not path.exists():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}


def _write_json(path: Path, data: Dict[str, Any], merge: bool = True) -> None:
  if not path.parent.exists():
    path.parent.mkdir(parents=True)

  final_data = data
  if merge and path.exists():
    # Simple merge of top-level keys
    existing = _read_json(path)
    # Deep merge special keys
    if "mappings" in data and "mappings" in existing:
      existing["mappings"].update(data["mappings"])
      data.pop("mappings")
    if "templates" in data and "templates" in existing:
      existing["templates"].update(data["templates"])
      data.pop("templates")

    existing.update(data)
    final_data = existing

  path.write_text(json.dumps(final_data, indent=2, sort_keys=True), encoding="utf-8")
  print(f"✅ Updated {path.relative_to(ROOT_DIR)}")


# --- SPEC UPDATERS (HUB) ---


def update_specs() -> None:
  """Updates the Abstract Operation Definitions in k_*.json."""

  # 1. k_neural_net.json
  nn_spec = {
    "BatchNorm": {"description": "Batch Normalization.", "std_args": ["input", "eps"]},
    "Pad": {"description": "Tensor padding.", "std_args": ["input", "pad", "mode", "value"]},
    "OneHot": {"description": "One-hot encoding.", "std_args": ["input", "num_classes"]},
    "CrossEntropyLoss": {"description": "Cross Entropy Loss.", "std_args": ["input", "target", "weight"]},
    "MSELoss": {"description": "Mean Squared Error.", "std_args": ["input", "target"]},
  }
  _write_json(SEMANTICS_DIR / "k_neural_net.json", nn_spec)

  # 2. k_array_api.json
  arr_spec = {
    "Clamp": {"description": "Clip values.", "std_args": ["input", "min", "max"]},
    "Gather": {"description": "Gather values.", "std_args": ["input", "dim", "index"]},
    "Scatter": {"description": "Scatter values.", "std_args": ["input", "dim", "index", "src"]},
    "Flatten": {"description": "Flatten dimensions.", "std_args": ["input", "start_dim", "end_dim"]},
    "View": {"description": "Reshape alias.", "std_args": ["input", "shape"]},
    "Reshape": {"description": "Reshape array.", "std_args": ["x", "shape"]},
    "Squeeze": {"description": "Remove dims.", "std_args": ["input", "dim"]},
    "Unsqueeze": {"description": "Add dim.", "std_args": ["input", "dim"]},
    "TopK": {"description": "Find k largest.", "std_args": ["input", "k", "dim"]},
  }
  # Add Casting Ops
  for t in ["float", "double", "int", "long", "bool"]:
    arr_spec[f"Cast{t.capitalize()}"] = {"description": f"Cast to {t}", "std_args": []}

  _write_json(SEMANTICS_DIR / "k_array_api.json", arr_spec)

  # 3. k_optimization.json
  opt_spec = {
    "ClipGradNorm": {"std_args": ["parameters", "max_norm"]},
    "StepLR": {"std_args": ["optimizer", "step_size", "gamma"]},
    "CosineAnnealingLR": {"std_args": ["optimizer", "T_max"]},
  }
  _write_json(SEMANTICS_DIR / "k_optimization.json", opt_spec)

  # 4. k_functional.json
  func_spec = {
    "vmap": {"std_args": ["func", "in_axes", "out_axes"]},
    "grad": {"std_args": ["func", "argnums"]},
    "jit": {"std_args": ["func", "static_argnums"]},
  }
  _write_json(SEMANTICS_DIR / "k_functional.json", func_spec)


# --- MAPPING GENERATORS (SPOKES) ---


def gen_torch_mappings() -> None:
  """Generates PyTorch Overlays."""
  mappings = {}

  # Casting
  for t in ["float", "double", "int", "long", "bool"]:
    mappings[f"Cast{t.capitalize()}"] = {"api": f"torch.Tensor.{t}"}
    mappings[t] = {"api": f"torch.Tensor.{t}"}

  # Math/Array
  mappings.update(
    {
      "Clamp": {"api": "torch.clamp"},
      "clamp": {"api": "torch.clamp"},
      "Gather": {"api": "torch.gather"},
      "Scatter": {"api": "torch.Tensor.scatter_"},
      "Flatten": {"api": "torch.flatten"},
      "View": {"api": "torch.Tensor.view"},
      "Reshape": {"api": "torch.reshape"},
      "Squeeze": {"api": "torch.squeeze"},
      "Unsqueeze": {"api": "torch.unsqueeze"},
      "TopK": {"api": "torch.topk"},
      "topk": {"api": "torch.topk"},
      "Pad": {"api": "torch.nn.functional.pad"},
    }
  )

  # Neural
  mappings.update(
    {
      "BatchNorm": {"api": "torch.nn.BatchNorm2d"},
      "OneHot": {"api": "torch.nn.functional.one_hot"},
      "one_hot": {"api": "torch.nn.functional.one_hot"},
      "CrossEntropyLoss": {"api": "torch.nn.functional.cross_entropy"},
      "MSELoss": {"api": "torch.nn.functional.mse_loss"},
      "cross_entropy": {"api": "torch.nn.functional.cross_entropy"},
    }
  )

  # Functional / Extras
  mappings.update(
    {
      "vmap": {"api": "torch.vmap", "args": {"in_axes": "in_dims", "out_axes": "out_dims"}},
      "grad": {"api": "torch.func.grad"},
      "jit": {"api": "torch.compile"},
      "Compile": {"api": "torch.compile"},
      "Synchronize": {"api": "torch.cuda.synchronize"},
      "ClipGradNorm": {"api": "torch.nn.utils.clip_grad_norm_"},
      "clip_grad_norm_": {"api": "torch.nn.utils.clip_grad_norm_"},
      "StepLR": {"api": "torch.optim.lr_scheduler.StepLR"},
      "CosineAnnealingLR": {"api": "torch.optim.lr_scheduler.CosineAnnealingLR"},
    }
  )

  payload = {"__framework__": "torch", "mappings": mappings}
  _write_json(SNAPSHOTS_DIR / "torch_vlatest_map.json", payload)


def gen_jax_mappings() -> None:
  """Generates JAX/Flax Overlays."""
  mappings = {}

  # Casting (via Plugin)
  cast_map = {"api": "astype", "requires_plugin": "type_methods"}
  for t in ["float", "double", "int", "long", "bool"]:
    mappings[f"Cast{t.capitalize()}"] = cast_map
    mappings[t] = cast_map

  # Math/Array
  mappings.update(
    {
      "Clamp": {"api": "jax.numpy.clip", "args": {"min": "a_min", "max": "a_max", "input": "a"}},
      "Gather": {"api": "jax.numpy.take_along_axis", "requires_plugin": "gather_adapter"},
      "Scatter": {"api": "jax.ops.index_update", "requires_plugin": "scatter_indexer"},
      "Flatten": {"api": "jax.numpy.reshape", "requires_plugin": "flatten_range"},
      "View": {"api": "jax.numpy.reshape", "requires_plugin": "view_semantics"},
      "Reshape": {"api": "jax.numpy.reshape", "requires_plugin": "pack_shape_args"},
      "Squeeze": {"api": "jax.numpy.squeeze", "args": {"dim": "axis"}},
      "Unsqueeze": {"api": "jax.numpy.expand_dims", "args": {"dim": "axis"}},
      "TopK": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"},
      "Pad": {"api": "jax.numpy.pad", "requires_plugin": "padding_converter"},
    }
  )

  # Neural & Checkpoint
  mappings.update(
    {
      "BatchNorm": {"api": "flax.nnx.BatchNorm", "requires_plugin": "batch_norm_unwrap", "args": {"eps": "epsilon"}},
      "OneHot": {"api": "jax.nn.one_hot", "args": {"tensor": "x", "input": "x"}},
      "LoadStateDict": {"api": "KeyMapper.from_torch", "requires_plugin": "checkpoint_mapper"},
      "load_state_dict": {"api": "KeyMapper.from_torch", "requires_plugin": "checkpoint_mapper"},
    }
  )

  # Loss
  mappings.update(
    {
      "CrossEntropyLoss": {
        "api": "optax.softmax_cross_entropy_with_integer_labels",
        "requires_plugin": "loss_reduction",
        "args": {"input": "logits", "target": "labels"},
      },
      "MSELoss": {
        "api": "optax.l2_loss",
        "requires_plugin": "loss_reduction",
        "args": {"input": "predictions", "target": "targets"},
      },
    }
  )

  # Functional / Extras
  mappings.update(
    {
      "vmap": {"api": "jax.vmap", "args": {"func": "fun"}},
      "grad": {"api": "jax.grad", "args": {"func": "fun"}},
      "jit": {"api": "jax.jit", "args": {"func": "fun"}},
      "ClipGradNorm": {"api": "optax.clip_by_global_norm", "requires_plugin": "grad_clipper"},
      "StepLR": {"api": "optax.exponential_decay", "requires_plugin": "scheduler_rewire"},
      "CosineAnnealingLR": {"api": "optax.cosine_decay_schedule", "requires_plugin": "scheduler_rewire"},
      "step": {"api": "noop", "requires_plugin": "scheduler_step_noop"},  # Scheduler step
    }
  )

  payload = {"__framework__": "jax", "mappings": mappings}
  _write_json(SNAPSHOTS_DIR / "jax_vlatest_map.json", payload)


def gen_mlx_mappings() -> None:
  """Generates MLX Overlays."""
  mappings = {}

  mappings.update(
    {
      "Compile": {"api": "mlx.core.compile", "requires_plugin": "mlx_compiler"},
      "Synchronize": {"api": "mx.eval", "requires_plugin": "mlx_synchronize"},
    }
  )

  payload = {"__framework__": "mlx", "mappings": mappings}
  _write_json(SNAPSHOTS_DIR / "mlx_vlatest_map.json", payload)


def main() -> None:
  print("🚀 Regenerating Semantic Knowledge Base...")
  update_specs()

  print("\n🛠️  Applying Framework Overlays...")
  gen_torch_mappings()
  gen_jax_mappings()
  gen_mlx_mappings()

  print("\n✨ Done.")


if __name__ == "__main__":
  main()
