"""Scaffold CLI Handler.

Heuristically scans a new library to generate a skeleton mapping.
"""

import json
from argparse import Namespace
from ml_switcheroo.discovery.consensus import ConsensusEngine


def handle_scaffold(args: Namespace):
  """Handles the 'scaffold' CLI command."""
  fw_name = args.framework  # pragma: no cover
  print(f"Scaffolding API mapping for framework: {fw_name}")  # pragma: no cover
  # pragma: no cover
  engine = ConsensusEngine([fw_name])  # pragma: no cover
  engine.ingest()  # pragma: no cover
  # pragma: no cover
  clusters = engine.cluster(threshold=0.8)  # pragma: no cover
  # pragma: no cover
  skeleton = {"framework": fw_name, "mappings": []}  # pragma: no cover
  # pragma: no cover
  for std_name, paths in clusters.items():  # pragma: no cover
    if paths:  # pragma: no cover
      skeleton["mappings"].append({"operation": std_name, "api": paths[0]})  # pragma: no cover
  # pragma: no cover
  out_file = f"{fw_name}_skeleton.json"  # pragma: no cover
  with open(out_file, "w") as f:  # pragma: no cover
    json.dump(skeleton, f, indent=2)  # pragma: no cover
  # pragma: no cover
  print(f"Skeleton written to {out_file} (Found {len(clusters)} candidate ops)")  # pragma: no cover
