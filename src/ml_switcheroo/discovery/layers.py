"""
Layer Discovery Subsystem (The Orchestrator).

This module provides the :class:`LayerDiscoveryBot`, which replaces the old hardcoded
polyfill logic. Instead of manually importing ``torch`` or ``flax`` and checking specific lists,
this bot queries the **Framework Registry**.

It iterates over every installed framework adapter, asks it to ``collect_api()``,
and feeds those results into the :class:`ConsensusEngine`. This allows the system to
automagically "learn" that ``mlx.nn.Linear`` corresponds to ``torch.nn.Linear`` without
explicitly programming that relationship here.

Usage:
    >>> from ml_switcheroo.discovery.layers import LayerDiscoveryBot
    >>> bot = LayerDiscoveryBot()
    >>> count = bot.run(dry_run=False)
"""

from collections import defaultdict
from typing import Dict, List, Any

# Core Imports
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.frameworks.base import StandardCategory, get_adapter
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.utils.console import log_info, log_success, log_warning
from ml_switcheroo.semantics import paths


class LayerDiscoveryBot:
  """
  Automated discovery bot for Neural Layers and Array Operations.

  This class orchestrates the "Consensus Discovery" workflow. It is completely
  agnostic to specific libraries; it relies exclusively on the
  :class:`FrameworkAdapter` protocol to abstract away framework introspection.
  """

  def __init__(self):
    """Initializes the discovery bot with consensus and persistence engines."""
    self.consensus = ConsensusEngine()
    self.persister = SemanticPersister()

  def run(self, dry_run: bool = False) -> int:
    """
    Executes the full discovery process across ALL registered, active frameworks.

    Steps:

    1.  **Harvest**: Iterates specifically over installed frameworks and requests
        their API surfaces via :meth:`collect_api` (Layers, Activations, Losses).
    2.  **Cluster**: Uses the Consensus Engine to group similar concepts
        (e.g., matching ``HuberLoss``, ``huber_loss``, and ``Huber``).
    3.  **Filter**: Discards candidates that don't satisfy the minimum
        support threshold (must exist in at least 2 frameworks to be a Standard).
    4.  **Align**: Determines standard argument names based on voting (e.g.,
        if 3 frameworks use ``axis`` and 1 uses ``dim``, ``axis`` wins).
    5.  **Persist**: Writes findings to ``semantics/k_discovered.json``.

    Args:
        dry_run (bool): If True, performs analysis and logging but does not
                        write to disk. Defects to False.

    Returns:
        int: The number of new abstract standards identified and staged.
    """
    log_info("🔍 Starting Dynamic Consensus Discovery...")

    # Dictionary mapping 'framework_key' -> List[GhostRef]
    inputs: Dict[str, List[GhostRef]] = defaultdict(list)

    # 1. Dynamic Framework Iteration
    # We query the registry for every known adapter (torch, jax, tensorflow, mlx, etc.)
    fws = available_frameworks()

    # Categories to scan. We cast a wide net to build a robust graph.
    scan_categories = [
      StandardCategory.LAYER,
      StandardCategory.ACTIVATION,
      StandardCategory.LOSS,
      StandardCategory.OPTIMIZER,
    ]

    for fw_name in fws:
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      # Accumulator for this specific framework
      fw_refs = []

      for cat in scan_categories:
        try:
          # Polymorphic call: The adapter decides if it scrapes live modules
          # (via inspect) or reads from a cached JSON snapshot (Ghost Mode).
          refs = adapter.collect_api(cat)
          if refs:
            fw_refs.extend(refs)
        except Exception:
          # Frameworks that are registered but not installed in the environment
          # will gracefully fail here if snapshots are missing.
          pass

      # Only register the framework if it actually contributed data
      if fw_refs:
        inputs[fw_name] = fw_refs
        log_info(f"  - {fw_name}: Found {len(fw_refs)} candidates.")

    if len(inputs) < 2:
      log_warning("Need at least 2 active frameworks to form consensus. Check installations.")
      return 0

    # 2. Consensus Clustering
    log_info("  - Clustering API surfaces...")
    candidates = self.consensus.cluster(inputs)

    # Filter: Keep if present in at least 2 frameworks (Intersection).
    # This prevents creating garbage standards that only exist in 1 library.
    valid = self.consensus.filter_common(candidates, min_support=2)

    if not valid:
      log_warning("No common operations found between frameworks.")
      return 0

    # 3. Align Signatures (Infer common args like 'dim' vs 'axis')
    self.consensus.align_signatures(valid)

    # 4. Refinement: Filter out weak candidates
    # e.g., Names with < 3 chars are often noisy aliases or variables like 'T', 'X'
    refined = [c for c in valid if len(c.name) > 2]

    if not refined:
      log_info("  No high-quality candidates passed refinement.")
      return 0

    detected_names = [c.name for c in refined[:5]]
    suffix = "..." if len(refined) > 5 else ""
    log_info(f"  Examples: {', '.join(detected_names)}{suffix}")

    if dry_run:
      log_info(f"  [Dry] Would persist {len(refined)} standards.")
      return len(refined)

    # 5. Persistence
    # We write to k_discovered.json to separate machine-learned knowledge from manual specs
    sem_dir = paths.resolve_semantics_dir()
    if not sem_dir.exists():
      sem_dir.mkdir(parents=True, exist_ok=True)

    target = sem_dir / "k_discovered.json"

    self.persister.persist(refined, target)

    log_success(f"✅ Discovery Complete. Mapped {len(refined)} new standards to {target.name}")
    return len(refined)
