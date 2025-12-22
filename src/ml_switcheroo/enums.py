"""
Enumerations for ml-switcheroo.

This module defines standard enumerations used across the codebase for
semantic categorization and framework identification.
"""

from enum import Enum


class SemanticTier(str, Enum):
  """
  Categorization of API operations to distinct knowledge base tiers.

  Used to route definitions to specific JSON files in `src/ml_switcheroo/semantics/`.

  Attributes:
      ARRAY_API: Basic math and array manipulation (e.g. `abs`, `sum`).
                 Maps to `k_array_api.json`.
      NEURAL: Nural network layers and stateful operations (e.g. `Conv2d`, `Linear`).
              Maps to `k_neural_net.json`.
      EXTRAS: Framework-specific utilities, IO, and device management.
              Maps to `k_framework_extras.json`.
  """

  ARRAY_API = "array"
  NEURAL = "neural"
  EXTRAS = "extras"


class LogicOp(str, Enum):
  """
  Supported operators for conditional logic rules in operations.
  Used for Conditional API Dispatch.
  """

  EQ = "eq"  # ==
  NEQ = "neq"  # !=
  GT = "gt"  # >
  LT = "lt"  # <
  GTE = "gte"  # >=
  LTE = "lte"  # <=
  IN = "in"  # value in [list]
  NOT_IN = "not_in"  # value not in [list]
