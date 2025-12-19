"""
Script to update 'snapshots/tensorflow_vlatest_map.json' with Data Loader support.

This maintenance script injects the specific mapping required to convert
`torch.utils.data.DataLoader` into `tf.data.Dataset` pipelines via the
`tf_data_loader` plugin.
"""

import json
from pathlib import Path


def update_tf_snapshot() -> None:
  """
  Reads the TensorFlow snapshot, injects DataLoader plugin mappings, and saves it.

  Updates:
  - Maps `DataLoader` to `tf.data.Dataset` (via `tf_data_loader` plugin).
  - Maps `TensorDataset` to `None` (it is unwrapped by the plugin logic).
  """
  snap_path = Path("src/ml_switcheroo/snapshots/tensorflow_vlatest_map.json")
  if not snap_path.exists():
    print("Skipping: Tensorflow snapshot not found.")
    return

  data = json.loads(snap_path.read_text())

  # Add DataLoader mapping
  # Note: 'api' field is somewhat dummy here as the plugin rewrites the whole chain,
  # but we point to tf.data.Dataset for reference.
  data["mappings"]["DataLoader"] = {"api": "tf.data.Dataset", "requires_plugin": "tf_data_loader"}

  # Ensure dependencies (TensorDataset) are mapped to tuple or ignored
  # We map it to None to let it pass through arguments, which our plugin extracts.
  data["mappings"]["TensorDataset"] = None

  snap_path.write_text(json.dumps(data, indent=2))
  print(f"✅ Updated {snap_path} with tf_data_loader plugin.")


if __name__ == "__main__":
  update_tf_snapshot()
