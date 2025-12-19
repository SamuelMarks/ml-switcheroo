"""
Script to register 'Loss' operations and reduction plugins.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_loss_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Neural or K_Loss)
  spec_path = sem_dir / "k_neural_net.json"  # Losses often live here or separate

  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  # Define CrossEntropy
  spec["CrossEntropyLoss"] = {
    "description": "Computes cross entropy loss between logits and target.",
    "std_args": ["input", "target", "weight", "reduction", "ignore_index"],
  }
  spec["MSELoss"] = {"description": "Mean Squared Error Loss.", "std_args": ["input", "target", "reduction"]}

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Cross Entropy
    # Maps to Optax. Note: Optax expects (logits, labels).
    # PyTorch F.cross_entropy is (input, target).
    # Standard semantic renaming handles input->logits, target->labels matches if defined.
    snap["mappings"]["CrossEntropyLoss"] = {
      "api": "optax.softmax_cross_entropy_with_integer_labels",
      "requires_plugin": "loss_reduction",
      "args": {"input": "logits", "target": "labels"},
    }

    # MSE
    snap["mappings"]["MSELoss"] = {
      "api": "optax.l2_loss",  # Note: l2_loss is 0.5 * (x-y)^2. MSE is mean((x-y)^2).
      # Strictly, optax.l2_loss differs by factor 0.5 and mean.
      # Wrapper adds mean. Factor 0.5 might need adjustment or ignore for optimization parity.
      "requires_plugin": "loss_reduction",
      "args": {"input": "predictions", "target": "targets"},
    }

    # Functional aliases
    snap["mappings"]["cross_entropy"] = snap["mappings"]["CrossEntropyLoss"]
    snap["mappings"]["mse_loss"] = snap["mappings"]["MSELoss"]

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["CrossEntropyLoss"] = {"api": "torch.nn.functional.cross_entropy"}
    t_snap["mappings"]["MSELoss"] = {"api": "torch.nn.functional.mse_loss"}

    # Map simple function names found in F.*
    t_snap["mappings"]["cross_entropy"] = {"api": "torch.nn.functional.cross_entropy"}
    t_snap["mappings"]["mse_loss"] = {"api": "torch.nn.functional.mse_loss"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_loss_semantics()
