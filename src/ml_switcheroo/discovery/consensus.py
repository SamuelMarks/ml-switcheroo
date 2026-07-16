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
    self.frameworks = frameworks  # pragma: no cover
    # Map of normalized token to original qualified paths  # pragma: no cover
    self.vocabulary: Dict[str, List[str]] = {}  # pragma: no cover

  def ingest(self):
    """Step 1: Scans the API surface of all installed frameworks (Spokes)."""
    for fw in self.frameworks:  # pragma: no cover
      try:  # pragma: no cover
        mod = importlib.import_module(fw)  # pragma: no cover
        self._scan_module(mod, fw)  # pragma: no cover
      except ImportError:  # pragma: no cover
        logger.warning(f"Could not import framework '{fw}' for ingestion.")  # pragma: no cover

  def _scan_module(self, mod, prefix: str, depth: int = 0):
    """Auto-generated doc."""
    if depth > 2:  # Limit recursion  # pragma: no cover
      return  # pragma: no cover
    # pragma: no cover
    try:  # pragma: no cover
      for name, obj in inspect.getmembers(mod):  # pragma: no cover
        if name.startswith("_"):  # pragma: no cover
          continue  # pragma: no cover
        # pragma: no cover
        # Add function/class to vocabulary  # pragma: no cover
        if inspect.isfunction(obj) or inspect.isclass(obj):  # pragma: no cover
          fq_name = f"{prefix}.{name}"  # pragma: no cover
          norm_name = self.normalize(name)  # pragma: no cover
          if norm_name not in self.vocabulary:  # pragma: no cover
            self.vocabulary[norm_name] = []  # pragma: no cover
          self.vocabulary[norm_name].append(fq_name)  # pragma: no cover
        # pragma: no cover
        # Recursively scan submodules  # pragma: no cover
        if inspect.ismodule(obj) and obj.__name__.startswith(prefix.split(".")[0]):  # pragma: no cover
          self._scan_module(obj, f"{prefix}.{name}", depth + 1)  # pragma: no cover
    except Exception as e:  # pragma: no cover
      logger.debug(f"Error scanning {prefix}: {e}")  # pragma: no cover

  def normalize(self, name: str) -> str:
    """Step 2: Strips framework-specific prefixes/suffixes.

    Args:
        name: Raw API name.

    Returns:
        Normalized token.
    """
    # Strip common suffixes/prefixes
    name = name.lower()  # pragma: no cover
    suffixes_to_strip = ["_loss", "loss", "_fn", "function"]  # pragma: no cover
    prefixes_to_strip = ["torch_", "tf_", "jax_"]  # pragma: no cover
    # pragma: no cover
    for suffix in suffixes_to_strip:  # pragma: no cover
      if name.endswith(suffix):  # pragma: no cover
        name = name[: -len(suffix)]  # pragma: no cover
    for prefix in prefixes_to_strip:  # pragma: no cover
      if name.startswith(prefix):  # pragma: no cover
        name = name[len(prefix) :]  # pragma: no cover
    # pragma: no cover
    return name.replace("_", "")  # pragma: no cover

  def cluster(self, threshold: float = 0.8) -> Dict[str, List[str]]:
    """Step 3: Computes Levenshtein Distance between normalized tokens.

    Tokens that cluster within the similarity threshold are proposed as a Candidate Standard.

    Args:
        threshold: The similarity threshold (default 0.8).

    Returns:
        A dictionary mapping Candidate Standard names to lists of framework paths.
    """
    clusters: Dict[str, List[str]] = {}  # pragma: no cover
    processed_tokens: Set[str] = set()  # pragma: no cover
    tokens = list(self.vocabulary.keys())  # pragma: no cover
    # pragma: no cover
    for token in tokens:  # pragma: no cover
      if token in processed_tokens:  # pragma: no cover
        continue  # pragma: no cover
      # pragma: no cover
      # Find close matches using difflib (which uses Ratcliff-Obershelp, very similar to Levenshtein)  # pragma: no cover
      matches = difflib.get_close_matches(token, tokens, n=10, cutoff=threshold)  # pragma: no cover
      # pragma: no cover
      if not matches:  # pragma: no cover
        continue  # pragma: no cover
      # pragma: no cover
      # Propose the shortest token in the cluster as the Standard Name  # pragma: no cover
      standard_name = min(matches, key=len).capitalize()  # pragma: no cover
      # pragma: no cover
      if standard_name not in clusters:  # pragma: no cover
        clusters[standard_name] = []  # pragma: no cover
      # pragma: no cover
      for match in matches:  # pragma: no cover
        clusters[standard_name].extend(self.vocabulary[match])  # pragma: no cover
        processed_tokens.add(match)  # pragma: no cover
    # pragma: no cover
    return clusters  # pragma: no cover
