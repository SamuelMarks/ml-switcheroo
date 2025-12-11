"""
Consensus Engine: The Dynamic Standardization Brain.

This module implements the logic to automatically discover Abstract Standards
by analyzing API lists from multiple frameworks. It identifies common concepts
(e.g., "HuberLoss" in Torch vs "huber" in JAX) via fuzzy clustering and naming
normalization, proposing them as "Candidate Standards" for the Middle Layer.

It also performs "Signature Alignment" to determine the standard arguments
(e.g. deciding that 'lr' is the standard argument for learning rate across frameworks).
"""

from typing import Dict, List, Optional, Set, Counter
from pydantic import BaseModel, Field

from ml_switcheroo.core.ghost import GhostRef


class CandidateStandard(BaseModel):
  """
  A proposed Abstract Standard discovered via consensus.

  Represents a single concept (e.g., 'Huber') found in one or more frameworks.

  Attributes:
      name: The proposed Abstract Name (e.g. 'Huber').
      variants: Map of {framework_name: GhostRef} for the implementations found.
      score: A confidence score (0.0-1.0) based on how many frameworks agree.
      std_args: List of abstract argument names agreed upon by consensus.
      arg_mappings: Nested dictionary {framework_name: {std_arg: fw_arg}}.
                    Used to generate the keys in the final StandardMap.
  """

  name: str
  variants: Dict[str, GhostRef] = Field(default_factory=dict)
  score: float = 0.0
  std_args: List[str] = Field(default_factory=list)
  arg_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)

  def add_variant(self, framework: str, ref: GhostRef):
    """Registers a framework's implementation of this standard."""
    self.variants[framework] = ref
    # Simple score based on number of votes
    self.score = float(len(self.variants))


class ConsensusEngine:
  """
  Algorithms for aligning divergent API naming conventions.

  Responsibility:
  1. Normalize API names (Clustering).
  2. Normalize Argument names (Signature Alignment).
  3. Compute intersection of arguments to define the Standard.
  """

  # Suffixes that carry framework implementation details rather than semantic meaning
  IGNORED_SUFFIXES = [
    "loss",
    "error",
    "layer",
    "block",
    "2d",
    "1d",
    "3d",
    "v1",
    "v2",
    "object",
  ]

  # Map of {SpecificVariant: CanonicalKey}
  # Used to align 'learning_rate' (JAX) with 'lr' (Torch)
  ARG_ALIASES = {
    "learning_rate": "lr",
    "rate": "lr",
    "axis": "dim",
    "dimension": "dim",
    "epsilon": "eps",
    "keepdims": "keep_dim",
    "weights": "weight",  # Torch often uses 'weight', others 'weights'
    "prob": "p",
    "probability": "p",
    "kernel_shape": "kernel_size",
    "filters": "out_channels",  # Keras vs Torch convention
    "features": "out_features",  # Flax vs Torch
  }

  @classmethod
  def normalize_name(cls, name: str) -> str:
    """
    Reduces an API Name to its semantic core for comparison.
    """
    # 1. Lowercase
    normalized = name.lower()

    # 2. Remove camelCase/Snake_case separators
    normalized = normalized.replace("_", "")

    # 3. Strip common noise suffixes
    clean = False
    while not clean:
      clean = True
      for suffix in cls.IGNORED_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
          normalized = normalized[: -len(suffix)]
          clean = False

    return normalized

  @classmethod
  def normalize_arg(cls, arg_name: str) -> str:
    """
    Canonicalizes an argument name.
    e.g., 'learning_rate' -> 'lr'.
    """
    lower = arg_name.lower()
    return cls.ARG_ALIASES.get(lower, lower)

  def cluster(self, framework_inputs: Dict[str, List[GhostRef]]) -> List[CandidateStandard]:
    """
    Groups API definitions from multiple frameworks into Candidates.
    """
    clusters: Dict[str, CandidateStandard] = {}

    for fw_name, distinct_refs in framework_inputs.items():
      for ref in distinct_refs:
        key = self.normalize_name(ref.name)

        if not key:
          key = ref.name.lower()

        if key not in clusters:
          abstract_name = key.capitalize()
          clusters[key] = CandidateStandard(name=abstract_name)

        clusters[key].add_variant(fw_name, ref)

    results = list(clusters.values())
    results.sort(key=lambda x: x.score, reverse=True)
    return results

  def filter_common(self, candidates: List[CandidateStandard], min_support: int = 2) -> List[CandidateStandard]:
    """
    Returns only standards derived from agreement between multiple frameworks.
    """
    return [c for c in candidates if len(c.variants) >= min_support]

  def align_signatures(self, candidates: List[CandidateStandard], consensus_threshold: float = 0.5):
    """
    Analyses the signatures of all variants in a candidate.
    Populates 'std_args' and 'arg_mappings' based on consensus.

    Args:
        candidates: List of CandidateStandards to process (in-place modification).
        consensus_threshold: Fraction of variants that must share an arg for it to be standard.
                             (0.5 means > 50% must have it).
    """
    for cand in candidates:
      # 1. Harvest all args from all variants
      # Map: {canonical_arg: {fw_name: original_arg_name}}
      arg_matrix: Dict[str, Dict[str, str]] = {}

      total_variants = len(cand.variants)
      if total_variants == 0:
        continue

      for fw_name, ref in cand.variants.items():
        for param in ref.params:
          # Skip 'self' or varargs if ghost inspection didn't filter them (it usually does)
          if param.name == "self":
            continue

          canonical = self.normalize_arg(param.name)

          if canonical not in arg_matrix:
            arg_matrix[canonical] = {}

          arg_matrix[canonical][fw_name] = param.name

      # 2. Determine Consensus
      std_args = []
      mappings = {fw: {} for fw in cand.variants}

      for canonical, occurrences in arg_matrix.items():
        support = len(occurrences) / total_variants

        # If support strictly greater than threshold
        if support > consensus_threshold:
          std_args.append(canonical)

          # Populate mappings for frameworks that have this arg
          for fw, original_name in occurrences.items():
            mappings[fw][canonical] = original_name

      cand.std_args = std_args
      cand.arg_mappings = mappings
