import os

frameworks = ["jax", "keras", "tensorflow", "mlx", "flax_nnx"]
src_dir = "src/ml_switcheroo/frameworks"
dest_dir = "snapshot-extractor/src/ml_snapshots/frameworks"

for fw in frameworks:
  src_file = os.path.join(src_dir, f"{fw}.py")
  dest_file = os.path.join(dest_dir, f"{fw}.py")

  with open(src_file, "r") as f:
    content = f.read()

  # We just want the imports and the _scan_* functions and _collect_live.
  # It's actually easier to just copy the whole file but remove FrameworkAdapter inheritances.
  # But wait, these adapters also have convert(), apply_wiring(), etc.
