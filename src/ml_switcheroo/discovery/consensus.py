"""Consensus Engine for Automated Discovery.

Implements the "Cyborg Workflow" consensus algorithm described in the research paper.
It scans API surfaces, normalizes names, and clusters them using Levenshtein distance
to propose candidate standards for the ODL.
"""

import importlib
import inspect
import difflib
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class ConsensusEngine:
  """The Consensus Engine for automated API mapping discovery."""

  def __init__(self, frameworks: List[str]):
    """Initializes the engine with a list of framework names to scan.

    Args:
        frameworks: List of framework modules (e.g., ['torch', 'jax.numpy']).
    """
    self.frameworks = frameworks
    # Map of normalized token to original qualified paths
    self.vocabulary: Dict[str, List[str]] = {}

  def ingest(self):
    """Step 1: Scans the API surface of all installed frameworks (Spokes)."""
    for fw in self.frameworks:
      try:
        mod = importlib.import_module(fw)
        self._scan_module(mod, fw)
      except ImportError:
        logger.warning(f"Could not import framework '{fw}' for ingestion.")

  def _scan_module(self, mod, prefix: str, depth: int = 0):
    """Auto-generated doc."""
    if depth > 2:  # Limit recursion
      return

    try:
      for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
          continue

        # Add function/class to vocabulary
        if inspect.isfunction(obj) or inspect.isclass(obj):
          fq_name = f"{prefix}.{name}"
          norm_name = self.normalize(name)
          if norm_name not in self.vocabulary:
            self.vocabulary[norm_name] = []
          self.vocabulary[norm_name].append(fq_name)

        # Recursively scan submodules
        if inspect.ismodule(obj) and obj.__name__.startswith(prefix.split(".")[0]):
          self._scan_module(obj, f"{prefix}.{name}", depth + 1)
    except Exception as e:
      logger.debug(f"Error scanning {prefix}: {e}")

  def normalize(self, name: str) -> str:
    """Step 2: Strips framework-specific prefixes/suffixes.

    Args:
        name: Raw API name.

    Returns:
        Normalized token.
    """
    # Strip common suffixes/prefixes
    name = name.lower()
    suffixes_to_strip = ["_loss", "loss", "_fn", "function"]
    prefixes_to_strip = ["torch_", "tf_", "jax_"]

    for suffix in suffixes_to_strip:
      if name.endswith(suffix):
        name = name[: -len(suffix)]
    for prefix in prefixes_to_strip:
      if name.startswith(prefix):
        name = name[len(prefix) :]

    return name.replace("_", "")

  def cluster(self, threshold: float = 0.8) -> Dict[str, List[str]]:
    """Step 3: Computes Levenshtein Distance between normalized tokens.

    Tokens that cluster within the similarity threshold are proposed as a Candidate Standard.

    Args:
        threshold: The similarity threshold (default 0.8).

    Returns:
        A dictionary mapping Candidate Standard names to lists of framework paths.
    """
    clusters: Dict[str, List[str]] = {}
    processed_tokens: Set[str] = set()
    tokens = list(self.vocabulary.keys())

    for token in tokens:
      if token in processed_tokens:
        continue

      # Find close matches using difflib (which uses Ratcliff-Obershelp, very similar to Levenshtein)
      matches = difflib.get_close_matches(token, tokens, n=10, cutoff=threshold)

      if not matches:
        continue

      # Propose the shortest token in the cluster as the Standard Name
      standard_name = min(matches, key=len).capitalize()

      if standard_name not in clusters:  # pragma: no cover
        clusters[standard_name] = []

      for match in matches:
        clusters[standard_name].extend(self.vocabulary[match])
        processed_tokens.add(match)

    return clusters
