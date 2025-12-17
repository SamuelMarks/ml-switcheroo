"""
scripts/discover_missing_ops.py

A specialized discovery bot that scans installed ML frameworks for Neural Layers.
It acts as a polyfill for FrameworkAdapters that may not fully implement
generic Layer discovery yet, ensuring operations like 'Linear' and 'ReLU'
are registered in the Knowledge Base without manual entry.

Usage:
    python scripts/discover_missing_ops.py
"""

import sys
import inspect
import difflib
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Ensure we can import the src package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_switcheroo.core.ghost import GhostInspector, GhostRef
from ml_switcheroo.discovery.consensus import ConsensusEngine, CandidateStandard
from ml_switcheroo.semantics.autogen import SemanticPersister

# Import semantics.paths via module to facilitate patching
from ml_switcheroo.semantics import paths
from ml_switcheroo.utils.console import log_info, log_success, log_warning

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def scan_torch_layers() -> List[GhostRef]:
  """
  Manually scans torch.nn for Layer classes.
  """
  results = []
  try:
    import torch.nn

    # List of common layers to hunt for if full scan is too noisy
    targets = [
      "Linear",
      "Conv1d",
      "Conv2d",
      "Conv3d",
      "LSTM",
      "GRU",
      "RNN",
      "Dropout",
      "BatchNorm1d",
      "BatchNorm2d",
      "LayerNorm",
      "ReLU",
      "GELU",
      "Sigmoid",
      "Tanh",
      "Softmax",
      "Embedding",
      "Flatten",
    ]

    for name, obj in inspect.getmembers(torch.nn):
      if inspect.isclass(obj):
        # Heuristic: Match specific target names
        if name in targets:
          try:
            ref = GhostInspector.inspect(obj, f"torch.nn.{name}")
            results.append(ref)
          except Exception:
            pass
  except ImportError:
    log_warning("Could not import torch.nn. Skipping Torch scan.")

  return results


def scan_torch_functional() -> List[GhostRef]:
  """
  Manually scans torch.nn.functional for activation functions.
  Ensures lowercase 'relu' logic matches source code usage.
  """
  results = []
  try:
    import torch.nn.functional as F

    targets = [
      "relu",
      "gelu",
      "sigmoid",
      "tanh",
      "softmax",
      "log_softmax",
      "silu",
      "elu",
      "leaky_relu",
      "dropout",
    ]

    for name in targets:
      if hasattr(F, name):
        try:
          obj = getattr(F, name)
          # Ensure it is a function
          if callable(obj):
            ref = GhostInspector.inspect(obj, f"torch.nn.functional.{name}")
            results.append(ref)
        except Exception:
          pass
  except ImportError:
    pass

  return results


def scan_flax_layers() -> List[GhostRef]:
  """
  Manually scans Flax NNX for Layer classes.
  """
  results = []
  try:
    # Try importing nnx (Flax 3.x+ feature)
    try:
      from flax import nnx
    except ImportError:
      # Fallback for dynamic/mock environments where sys.modules has it
      if "flax.nnx" in sys.modules:
        from flax import nnx
      else:
        return []

    targets = [
      "Linear",
      "Conv",
      "Embed",
      "Dropout",
      "BatchNorm",
      "LayerNorm",
      "relu",
      "gelu",
      "sigmoid",
      "tanh",
      "softmax",
      "log_softmax",
      "silu",
      "elu",
      "leaky_relu",
    ]

    for name, obj in inspect.getmembers(nnx):
      # Check classes or functions (activations often functions)
      if inspect.isclass(obj) or inspect.isfunction(obj):
        if name in targets:
          try:
            ref = GhostInspector.inspect(obj, f"flax.nnx.{name}")
            results.append(ref)
          except Exception:
            pass
  except ImportError:
    log_warning("Could not import flax.nnx. Skipping Flax scan.")

  return results


def fuzzy_cluster_layers(inputs: Dict[str, List[GhostRef]], threshold: float = 0.8) -> List[CandidateStandard]:
  """
  Groups references from different frameworks using Levenshtein distance.
  e.g., Groups 'torch.nn.Linear' and 'flax.nnx.Linear' into 'Linear'.
  """
  engine = ConsensusEngine()
  clusters = {}

  # Flatten inputs
  all_refs = []
  for fw, refs in inputs.items():
    for ref in refs:
      all_refs.append((fw, ref))

  # Sort references by name length (descending) to prefer full names as cluster keys
  all_refs.sort(key=lambda x: len(x[1].name), reverse=True)

  for fw, ref in all_refs:
    # Normalize name (lowercase, strip suffixes like 'Layer')
    norm = engine.normalize_name(ref.name)

    match_key = None

    # 1. Exact Match on Normalized Key
    if norm in clusters:
      match_key = norm
    # 2. Fuzzy Match
    else:
      existing_keys = list(clusters.keys())
      matches = difflib.get_close_matches(norm, existing_keys, n=1, cutoff=threshold)
      if matches:
        match_key = matches[0]

    if match_key:
      clusters[match_key].add_variant(fw, ref)
      cand = clusters[match_key]

      # --- NAMING PREFERENCE LOGIC ---
      # If we find a lowercase function Ref 'relu', and current standard name is 'Relu' (Class),
      # we overwrite the Standard Name to 'relu' to prefer functional casing.
      # This aligns with JAX/NumPy conventions.
      if ref.kind == "function" and not cand.name[0].islower():
        cand.name = ref.name

      # Also if we explicitly found a 'Relu' functionRef but key was normalized?
      # Trust ref.name for exact casing if it looks better.

    else:
      # New Cluster
      # Use original name (e.g. "Linear") for display, stripped of path
      display_name = ref.name.split(".")[-1]

      # Capitalize if it looks like a layer class (kind='class')
      if ref.kind == "class" and len(display_name) > 0:
        display_name = display_name[0].upper() + display_name[1:]

      # If it comes from functional (e.g. F.relu -> 'relu'), preserve lowercase
      if ref.kind == "function":
        display_name = ref.name

      cand = CandidateStandard(name=display_name)
      cand.add_variant(fw, ref)
      clusters[norm] = cand

  return list(clusters.values())


def main():
  log_info("🔍 Starting Discovery of Missing Operations (Layers/Activations)...")

  inputs = defaultdict(list)

  # 1. Harvest Torch
  torch_layers = scan_torch_layers()
  torch_funcs = scan_torch_functional()
  torch_ops = torch_layers + torch_funcs
  if torch_ops:
    inputs["torch"].extend(torch_ops)
    log_info(f"  Found {len(torch_ops)} candidates in PyTorch.")

  # 2. Harvest Flax
  # We map this to 'flax_nnx' framework key used by adapters/tests
  flax_ops = scan_flax_layers()
  if flax_ops:
    inputs["flax_nnx"].extend(flax_ops)
    log_info(f"  Found {len(flax_ops)} candidates in Flax NNX.")

  if not inputs:
    log_warning("No APIs found. Ensure 'torch' and 'flax' are installed.")
    return

  # 3. Consensus
  candidates = fuzzy_cluster_layers(inputs)

  # Filter: Keep if present in at least 1 framework.
  # Just 1 framework is enough to create a spec entry that can be manually mapped later.
  engine = ConsensusEngine()
  valid = engine.filter_common(candidates, min_support=1)

  if not valid:
    log_warning("No candidates formed.")
    return

  # 4. Align Signatures (Infer args)
  engine.align_signatures(valid)

  # Debug print names
  detected_names = [c.name for c in valid]
  log_info(f"  Candidates identified: {', '.join(detected_names)}")

  # 5. Persist
  # Use 'paths.resolve' to allow clean patching in tests
  sem_dir = paths.resolve_semantics_dir()
  if not sem_dir.exists():
    sem_dir.mkdir(parents=True, exist_ok=True)

  target = sem_dir / "k_discovered.json"

  persister = SemanticPersister()
  persister.persist(valid, target)

  log_success(f"✅ Discovery Complete. Mapped {len(valid)} ops to {target.name}")


if __name__ == "__main__":
  main()
