"""
Meta Command Handlers.

This module provides handlers for introspection and schema export tools,
enabling external agents (like LLMs or IDEs) to understand the
internal data structures of ml-switcheroo.
"""

import json
from ml_switcheroo.core.dsl import OperationDef


def handle_schema() -> int:
  """
  Exports the Operation Definition Language (ODL) JSON Schema.

  Prints the JSON schema derived from the Pydantic model `OperationDef`
  to standard output. This schema defines the structure required for
  valid YAML inputs to the `define` command.

  Returns:
      int: Exit code (0 for success).
  """
  schema = OperationDef.model_json_schema()
  print(json.dumps(schema, indent=2))
  return 0
