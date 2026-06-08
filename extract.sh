#!/bin/bash
set -e

REPO_DIR="/Users/samuel/repos/ml-switcheroo/snapshot-extractor"

mkdir -p "$REPO_DIR/src/ml_snapshots/frameworks"
mkdir -p "$REPO_DIR/tests"

# Create pyproject.toml
cat << 'EOF' > "$REPO_DIR/pyproject.toml"
[project]
name = "ml-snapshots"
version = "0.1.0"
description = "Extracts API snapshots from ML frameworks for Ghost Mode operation"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
frameworks = [
    "torch>=2.0.0",
    "jax>=0.4.0",
    "tensorflow>=2.10.0",
    "keras>=3.0.0",
    "mlx>=0.10.0",
    "flax>=0.8.0",
]

[project.scripts]
ml-snapshots = "ml_snapshots.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF

# Create README.md
cat << 'EOF' > "$REPO_DIR/README.md"
# ML Snapshots

Extracts API snapshots from ML frameworks for Ghost Mode operation.
EOF

# Copy Ghost inspector models
cp src/ml_switcheroo/core/ghost.py "$REPO_DIR/src/ml_snapshots/models.py"

# Create enums.py
cat << 'EOF' > "$REPO_DIR/src/ml_snapshots/enums.py"
from enum import Enum

class StandardCategory(str, Enum):
    LOSS = "losses"
    OPTIMIZER = "optimizers"
    ACTIVATION = "activations"
    LAYER = "layers"
EOF

# Create __init__.py files
touch "$REPO_DIR/src/ml_snapshots/__init__.py"
touch "$REPO_DIR/src/ml_snapshots/frameworks/__init__.py"

# Next, we need a CLI to iterate over frameworks
cat << 'EOF' > "$REPO_DIR/src/ml_snapshots/cli.py"
import argparse
import json
import importlib.metadata
from pathlib import Path

from ml_snapshots.enums import StandardCategory
from ml_snapshots.frameworks.torch import collect_api as torch_collect
from ml_snapshots.frameworks.jax import collect_api as jax_collect
from ml_snapshots.frameworks.keras import collect_api as keras_collect
from ml_snapshots.frameworks.tensorflow import collect_api as tf_collect
from ml_snapshots.frameworks.mlx import collect_api as mlx_collect
from ml_snapshots.frameworks.flax_nnx import collect_api as flax_collect

FRAMEWORKS = {
    "torch": torch_collect,
    "jax": jax_collect,
    "keras": keras_collect,
    "tensorflow": tf_collect,
    "mlx": mlx_collect,
    "flax_nnx": flax_collect,
}

def get_pkg_version(package_name: str) -> str:
    try:
        if package_name == "flax_nnx":
            package_name = "flax"
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Capture ML Framework Snapshots")
    parser.add_argument("--out-dir", type=str, default="snapshots", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fw, collect_func in FRAMEWORKS.items():
        print(f"Scanning {fw}...")
        version = get_pkg_version(fw)
        if version == "unknown":
            print(f"Skipping {fw}, not installed.")
            continue

        snapshot_data = {"version": version, "categories": {}}
        found_any = False

        for category in StandardCategory:
            try:
                refs = collect_func(category)
                if refs:
                    refs.sort(key=lambda x: x.name)
                    found_any = True
                    snapshot_data["categories"][category.value] = [r.model_dump(exclude_unset=True) for r in refs]
            except Exception as e:
                print(f"Failed collecting {category.value} for {fw}: {e}")

        if found_any:
            safe_ver = version.replace("+", "_").replace(" ", "_")
            out_path = out_dir / f"{fw}_v{safe_ver}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2, sort_keys=True)
            print(f"Saved snapshot to {out_path}")

if __name__ == "__main__":
    main()
EOF

# Implement torch snapshot extraction
cat << 'EOF' > "$REPO_DIR/src/ml_snapshots/frameworks/torch.py"
import inspect
from typing import List
from ml_snapshots.models import GhostInspector, GhostRef
from ml_snapshots.enums import StandardCategory

try:
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    nn = None
    optim = None

def _scan_losses() -> List[GhostRef]:
    if not nn: return []
    found = []
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
            if issubclass(obj, nn.Module):
                found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

def _scan_optimizers() -> List[GhostRef]:
    if not optim: return []
    found = []
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and name != "Optimizer":
            try:
                if issubclass(obj, optim.Optimizer):
                    found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
            except TypeError:
                pass
    return found

def _scan_activations() -> List[GhostRef]:
    if not nn: return []
    found = []
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            if name in ["ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "Softmax", "LeakyReLU"]:
                found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

def _scan_layers() -> List[GhostRef]:
    if not nn: return []
    found = []
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            if not name.endswith("Loss") and name not in ["ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "Softmax", "LeakyReLU"]:
                found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

def collect_api(category: StandardCategory) -> List[GhostRef]:
    if category == StandardCategory.LOSS:
        return _scan_losses()
    elif category == StandardCategory.OPTIMIZER:
        return _scan_optimizers()
    elif category == StandardCategory.ACTIVATION:
        return _scan_activations()
    elif category == StandardCategory.LAYER:
        return _scan_layers()
    return []
EOF

echo "Done extracting core framework code to snapshot-extractor."
