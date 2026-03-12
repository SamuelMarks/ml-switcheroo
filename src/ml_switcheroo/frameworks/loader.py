"""
Framework Definition Loader.

This module provides utilities to load static operation definitions from JSON files
located in `src/ml_switcheroo/frameworks/definitions/`. It utilizes caching to
ensure efficient access during runtime and discovery.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

from ml_switcheroo.frameworks.base import StandardMap

# Locate the definitions directory relative to this file
# .../src/ml_switcheroo/frameworks/definitions
DEFINITIONS_DIR = Path(__file__).parent / "definitions"


@lru_cache(maxsize=None)
def load_definitions(framework: str) -> Dict[str, StandardMap]:
  """
  Loads static definitions for a specific framework from its JSON file.

  Uses `functools.lru_cache` to ensure the file is read and parsed only once per execution.
  Converts raw JSON dictionaries into `StandardMap` Pydantic objects.

  Args:
      framework (str): The framework key (e.g., 'torch', 'jax').

  Returns:
      Dict[str, StandardMap]: A dictionary mapping Operation Names to StandardDefs.
      Returns an empty dict if the definition file does not exist.
  """
  file_path = DEFINITIONS_DIR / f"{framework}.json"

  if not file_path.exists():
    return {}

  try:
    with open(file_path, "r", encoding="utf-8") as f:
      raw_data = json.load(f)

    # Convert dictionary entries to Pydantic models
    return {op_name: StandardMap.model_validate(op_def) for op_name, op_def in raw_data.items()}
  except (json.JSONDecodeError, OSError) as e:
    # Log via print/logging system in real app; here we return empty safely
    print(f"Failed to load definitions for {framework}: {e}")
    return {}


def clear_definition_cache() -> None:
  """
  Clears the LRU cache for definitions.
  Useful for tests or hot-reloading scenarios.
  """
  load_definitions.cache_clear()


def get_definitions_path(framework: str) -> Path:
  """
  Returns the resolved path for a framework's definition JSON.

  Args:
      framework (str): The framework key.

  Returns:
      Path: The absolute path to the intended JSON file.
  """
  return DEFINITIONS_DIR / f"{framework}.json"
