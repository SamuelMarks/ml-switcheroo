"""Framework Definition Loader.

This module provides utilities to load static operation definitions from JSON files
located in `ml_ecosystem_snapshots.snapshots` (or legacy `ml_framework_snapshots.snapshots`).
It utilizes caching to ensure efficient access during runtime and discovery.
"""

import importlib.resources
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from ml_switcheroo_ir.schema.ghost import StandardMap

logger = logging.getLogger(__name__)

try:
  from ml_ecosystem_snapshots._alias import register_alias_finder

  register_alias_finder()
except ImportError:
  pass


def _resolve_resource_file(framework: str) -> Any:
  """Resolve snapshot resource file across ecosystem and legacy packages.

  Args:
      framework (str): The framework name key.

  Returns:
      Any: A traversable resource file path for the framework JSON.
  """
  for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
    try:
      fp = importlib.resources.files(pkg).joinpath(f"{framework}.json")
      if fp.is_file():
        return fp
    except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError):
      pass
  try:
    return importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath(f"{framework}.json")
  except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError):
    return importlib.resources.files("ml_framework_snapshots.snapshots").joinpath(f"{framework}.json")


@lru_cache(maxsize=None)
def load_definitions(framework: str) -> Dict[str, StandardMap]:
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
  # Fallback to local path representation for testing/compatibility
  for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
    try:
      fp = importlib.resources.files(pkg).joinpath(f"{framework}.json")
      if fp.is_file():
        return Path(str(fp))
    except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError, Exception):
      pass
  try:
    return Path(str(importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath(f"{framework}.json")))
  except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError, Exception):
    try:
      return Path(str(importlib.resources.files("ml_framework_snapshots.snapshots").joinpath(f"{framework}.json")))
    except (ImportError, ModuleNotFoundError, OSError, TypeError, RuntimeError, Exception):
      return Path(f"{framework}.json")
