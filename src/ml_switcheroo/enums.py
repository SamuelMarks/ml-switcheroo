from enum import Enum

# NOTE: SupportedEngine Enum has been removed in favor of dynamic discovery.
# Frameworks are now registered via src/ml_switcheroo/frameworks/*.py
# and queried using `ml_switcheroo.frameworks.available_frameworks()`.


class SemanticTier(str, Enum):
  """
  Categorization of API operations to distinct JSON lookup files.
  """

  ARRAY_API = "array"  # Pure Math (semantics/k_array_api.json)
  NEURAL = "neural"  # Layers/Stateful (semantics/k_neural_net.json)
  EXTRAS = "extras"  # Framework Utilities (semantics/k_framework_extras.json)
