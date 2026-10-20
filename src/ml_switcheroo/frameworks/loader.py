"""Framework Definition Loader.

This module provides utilities to load static operation definitions from JSON files
located in `ml_ecosystem_snapshots.snapshots`.
It utilizes caching to ensure efficient access during runtime and discovery.
"""

import importlib.resources
import importlib.resources.abc
import json
import logging
from functools import lru_cache
from pathlib import Path
import typing

from ml_switcheroo_ir.schema.ghost import StandardMap

logger = logging.getLogger(__name__)


def _resolve_resource_file(framework: str) -> importlib.resources.abc.Traversable:
  """Resolve snapshot resource file from ml_ecosystem_snapshots.snapshots.

  Args:
      framework (str): The framework name key.

  Returns:
      importlib.resources.abc.Traversable: A traversable resource file path for the framework JSON.
  """
  return importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath(f"{framework}.json")


@lru_cache(maxsize=None)
def load_definitions(framework: str) -> typing.Dict[str, StandardMap]:
  """Load static definitions for a specific framework from its JSON file.

  Uses `functools.lru_cache` to ensure the file is read and parsed only once per execution.
  Converts raw JSON dictionaries into `StandardMap` Pydantic objects.

  Args:
      framework (str): The framework key (e.g., 'torch', 'jax').

  Returns:
      Dict[str, StandardMap]: A dictionary mapping Operation Names to StandardDefs.
      Returns an empty dict if the definition file does not exist.

  """
  try:
    file_path = _resolve_resource_file(framework)
  except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError):
    return {}
  if not file_path.is_file():
    return {}
  try:
    with file_path.open("r", encoding="utf-8") as f:
      raw_data = json.load(f)
    return {op_name: StandardMap.model_validate(op_def) for op_name, op_def in raw_data.items()}
  except (json.JSONDecodeError, OSError) as e:
    logger.warning("Failed to load definitions for %s: %s", framework, e)
    return {}


def clear_definition_cache() -> None:
  """Clear the LRU cache for definitions.

  Useful for tests or hot-reloading scenarios.
  """
  load_definitions.cache_clear()


def get_definitions_path(framework: str) -> Path:
  """Return the resolved path for a framework's definition JSON.

  Args:
      framework (str): The framework key.

  Returns:
      Path: The absolute path to the intended JSON file.

  """
  try:
    fp = importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath(f"{framework}.json")
    return Path(str(fp))
  except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError, Exception):
    return Path(f"{framework}.json")
