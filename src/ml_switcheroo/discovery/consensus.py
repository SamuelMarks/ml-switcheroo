"""
Consensus Engine: The Dynamic Standardization Brain.

This module provides the logic to identify common operations across disjoint frameworks.
It aligns variable naming conventions (e.g., 'dim' vs 'axis') and clusters API endpoints
(e.g., 'torch.sin' and 'jax.numpy.sin') into unified Abstract Standards suitable for
the Semantic Knowledge Base.
"""

from typing import Dict, List, Optional, Set, Any, Union
from collections import Counter as CollectionCounter
from pydantic import BaseModel, Field

from ml_switcheroo.core.ghost import GhostRef


class CandidateStandard(BaseModel):
  """
  A proposed Abstract Standard discovered via consensus.
  """

  name: str = Field(description="The abstract name (e.g., 'Conv2d').")
  variants: Dict[str, GhostRef] = Field(
    default_factory=dict, description="A map of framework names to their specific implementations."
  )
  score: float = Field(0.0, description="A confidence score derived from the number of concurring frameworks.")
  std_args: List[Union[str, Dict[str, Any]]] = Field(
    default_factory=list, description="The list of argument names or definitions deemed 'standard' via voting."
  )
  arg_mappings: Dict[str, Dict[str, str]] = Field(
    default_factory=dict, description="Nested dictionary mapping {framework: {std_arg: fw_specific_arg}}."
  )

  def add_variant(self, framework: str, ref: GhostRef) -> None:
    """
    Registers a framework's implementation of this concept.
    Updates the consensus score based on support count.

    Args:
        framework (str): The framework identifier (e.g. 'torch').
        ref (GhostRef): The API reference object found in that framework.
    """
    self.variants[framework] = ref
    self.score = float(len(self.variants))


class ConsensusEngine:
  """
  Algorithms for aligning divergent API naming conventions.

  Capabilities:

  1.  **Clustering**: Groups APIs like `HuberLoss`, `huber_loss`, and `Huber` together.
  2.  **Normalization**: Strips common noise (prefixes/suffixes) to find the semantic root.
  3.  **Signature Alignment**: Builds a translation map for arguments (e.g., `keepdims` <-> `keep_dims`).
  4.  **Type Consensus**: Aggregates type hints found in source candidates to enrich the standard signature.
  """

  # Suffixes that carry framework-specific implementation details rather than semantic meaning.
  # We strip these during clustering to find the 'core' concept.
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
    "op",
    "func",
  ]

  # Enhanced Aliases for robust consensus across frameworks.
  # Maps variable names to a single canonical representation.
  ARG_ALIASES = {
    # Optimization
    "learning_rate": "lr",
    "rate": "lr",
    # Dimensions
    "axis": "dim",
    "dimension": "dim",
    "axes": "dim",
    "dim": "dim",
    # Math / Logic
    "epsilon": "eps",
    "keepdims": "keepdim",
    "keep_dims": "keepdim",
    "prob": "p",
    "probability": "p",
    "inverse": "inv",
    # Neural
    "weights": "weight",
    "kernel_shape": "kernel_size",
    "filters": "out_channels",
    "features": "out_features",
    "input": "x",
    "a": "x",
    "input_tensor": "x",
    "tensor": "x",
  }

  @classmethod
  def normalize_name(cls, name: str) -> str:
    """
    Reduces an API Name to its semantic core for comparison.

    This removes casing, underscores, and common prefixes/suffixes.

    Examples:
        * 'HuberLoss' -> 'huber'
        * 'reduce_mean' -> 'mean'
        * 'conv2d' -> 'conv'

    Args:
        name (str): The raw API name (e.g. 'CrossEntropyLoss').

    Returns:
        str: The normalized key (e.g. 'crossentropy').
    """
    normalized = name.lower().replace("_", "")

    # Common functional prefixes to strip
    for prefix in ["reduce", "math", "ops", "nn", "special", "functional"]:
      if normalized.startswith(prefix) and len(normalized) > len(prefix):
        normalized = normalized[len(prefix) :]

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
    Canonicalizes an argument name using the alias map.

    Example:
        'learning_rate' -> 'lr'

    Args:
        arg_name (str): The raw argument name.

    Returns:
        str: The canonical standard name.
    """
    lower = arg_name.lower()
    return cls.ARG_ALIASES.get(lower, lower)

  def cluster(self, framework_inputs: Dict[str, List[GhostRef]]) -> List[CandidateStandard]:
    """
    Groups API definitions from multiple frameworks into Candidates based on name similarity.

    Args:
        framework_inputs: Dictionary mapping 'framework_name' -> List of discovered GhostRefs.

    Returns:
        List[CandidateStandard]: A list of potential standards, sorted by descending score.
    """
    clusters: Dict[str, CandidateStandard] = {}

    for fw_name, distinct_refs in framework_inputs.items():
      for ref in distinct_refs:
        # Normalized key for clustering (finding matches)
        key = self.normalize_name(ref.name)

        if not key:
          key = ref.name.lower()

        if key not in clusters:
          # Use a capitalized version of the normalized key as the Abstract Name
          abstract_name = key.capitalize()
          clusters[key] = CandidateStandard(name=abstract_name)

        clusters[key].add_variant(fw_name, ref)

    results = list(clusters.values())
    # Sort by score (number of frameworks implementing it) descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results

  def filter_common(self, candidates: List[CandidateStandard], min_support: int = 2) -> List[CandidateStandard]:
    """
    Filters candidates to keep only those present in a minimum number of frameworks.

    This ensures we only create standards for concepts that are truly shared across
    ecosystems, avoiding framework-specific noise.

    Args:
        candidates (List[CandidateStandard]): List of candidates from clustering.
        min_support (int): Minimum number of different frameworks that must implement the op.

    Returns:
        List[CandidateStandard]: Filtered list of robust candidates.
    """
    return [c for c in candidates if len(c.variants) >= min_support]

  def align_signatures(self, candidates: List[CandidateStandard], consensus_threshold: float = 0.5) -> None:
    """
    Analyses the arguments of all variants in a candidate to determine Standard Arguments and Types.

    It populates `std_args` on the candidate by voting:
    1.  If an argument (normalized) appears in >50% of the implementations, it becomes part of the standard.
    2.  If type hints are available across the variants, it determines the consensus type and
        populates a rich argument definition (e.g. `{'name': 'x', 'type': 'int'}`) instead of a simple string.

    It also populates `arg_mappings` to translate between the Standard name and
    the specific framework name (e.g. Standard 'dim' -> Torch 'dim', Jax 'axis').

    Args:
        candidates (List[CandidateStandard]): List of CandidateStandards to process (in-place modification).
        consensus_threshold (float): Fraction of variants that must share an arg (0.0 - 1.0).
    """
    for cand in candidates:
      # Map: {canonical_arg: {fw_name: original_arg_name}}
      arg_matrix: Dict[str, Dict[str, str]] = {}
      # Map: {canonical_arg: List[str_type]} - collects observed types for voting
      type_matrix: Dict[str, List[str]] = {}

      total_variants = len(cand.variants)

      if total_variants == 0:
        continue

      # 1. Harvest all args from all variants
      for fw_name, ref in cand.variants.items():
        for param in ref.params:
          # Access param attributes
          # GhostRef params are GhostParam objects
          p_name = getattr(param, "name", param) if hasattr(param, "name") else param
          p_str = str(p_name)

          # Ignore common object-oriented instance arguments
          if p_str in ["self", "cls"]:
            continue

          # Use normalized name for consensus tracking
          canonical = self.normalize_arg(p_str)

          if canonical not in arg_matrix:
            arg_matrix[canonical] = {}
            type_matrix[canonical] = []

          arg_matrix[canonical][fw_name] = p_str

          # Collect Type Hint
          anno = getattr(param, "annotation", None)
          if anno and anno not in ("None", "Any", "<unrepresentable>", ""):
            type_matrix[canonical].append(anno)

      std_args = []
      mappings = {fw: {} for fw in cand.variants}

      # 2. Vote for consensus
      for canonical, occurrences in arg_matrix.items():
        support = len(occurrences) / total_variants

        # If support is sufficient, this arg becomes part of the standard
        if support > consensus_threshold:
          # Determine Type Consensus
          observed_types = type_matrix.get(canonical, [])
          final_type = None

          if observed_types:
            # Simple majority vote logic for types
            ctr = CollectionCounter(observed_types)
            most_common = ctr.most_common(1)
            if most_common:
              best_type, _count = most_common[0]
              final_type = best_type

          if final_type:
            std_args.append({"name": canonical, "type": final_type})
          else:
            std_args.append(canonical)

          # Create mappings for frameworks that possess this arg
          # e.g., if canonical is 'dim', store mapping 'dim' -> 'axis' for JAX
          for fw, original_name in occurrences.items():
            mappings[fw][canonical] = original_name

      cand.std_args = std_args
      cand.arg_mappings = mappings
