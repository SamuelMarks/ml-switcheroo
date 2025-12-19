"""
Script to register 'OneHot' encoding operations.

Addresses the mapping between:
1. PyTorch: `torch.nn.functional.one_hot(tensor, num_classes=-1)`
2. JAX: `jax.nn.one_hot(x, num_classes, ...)`

While the argument names `num_classes` overlap, the input argument names differ
(`tensor` vs `x`). This configuration ensures correct keyword mapping if inputs
are passed by name, and establishes the API translation.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_onehot_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Neural_Net)
  spec_path = sem_dir / "k_neural_net.json"
  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  spec["OneHot"] = {
    "description": "Takes LongTensor with index values and returns a tensor of one-hot encoded vectors.",
    "std_args": ["input", "num_classes"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Mapping definition
    # JAX args: x, num_classes, dtype, axis
    # Torch args: tensor, num_classes.
    # We explicitly map 'tensor' -> 'x' to handle potential keyword usage.
    mapping = {
      "api": "jax.nn.one_hot",
      "args": {
        "tensor": "x",
        "input": "x",  # Handling generic input name if standardized
      },
    }

    snap["mappings"]["OneHot"] = mapping
    # Map generic function name 'one_hot' often imported from F
    snap["mappings"]["one_hot"] = mapping

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["OneHot"] = {"api": "torch.nn.functional.one_hot"}
    t_snap["mappings"]["one_hot"] = {"api": "torch.nn.functional.one_hot"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_onehot_semantics()
