"""Scaffold CLI Handler.

Heuristically scans a new library to generate a skeleton mapping.
"""

from typing import Any

import json
from argparse import Namespace
from ml_switcheroo.discovery.consensus import ConsensusEngine


def handle_scaffold(args: Namespace) -> Any:
  """Handles the 'scaffold' CLI command."""
  fw_name = args.framework
  print(f"Scaffolding API mapping for framework: {fw_name}")

  engine = ConsensusEngine([fw_name])
  engine.ingest()

  clusters = engine.cluster(threshold=0.8)

  skeleton = {"framework": fw_name, "mappings": []}

  for std_name, paths in clusters.items():
    if paths:
      skeleton["mappings"].append({"operation": std_name, "api": paths[0]})

  out_file = f"{fw_name}_skeleton.json"
  with open(out_file, "w") as f:
    json.dump(skeleton, f, indent=2)

  print(f"Skeleton written to {out_file} (Found {len(clusters)} candidate ops)")
