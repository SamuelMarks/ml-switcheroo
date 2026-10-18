"""Enumeration for ml-switcheroo.

This module defines standard enumerations used across the codebase for
semantic categorization and framework identification.
"""

from enum import Enum

from ml_switcheroo_ir.schema.ghost import SemanticTier as SemanticTier


class LogicOp(str, Enum):
  """Supported operators for conditional logic rules in operations.

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
  IS_TYPE = "is_type"  # Checks if AST node matches type (int, list, etc)
