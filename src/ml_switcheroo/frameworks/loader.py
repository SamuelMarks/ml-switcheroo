"""Framework Definition Loader.

This module provides utilities to load static operation definitions from JSON files
located in `ml_ecosystem_snapshots.snapshots`.
It utilizes caching to ensure efficient access during runtime and discovery.
"""

import importlib.resources
import json
import logging
import typing
from functools import lru_cache
from pathlib import Path

from ml_switcheroo_ir.schema.ghost import StandardMap

logger = logging.getLogger(__name__)


class ResourceFileProtocol(typing.Protocol):
  """Structural protocol for traversable snapshot resource files across Python 3.10+."""

  def is_file(self) -> bool:
    """Return True if the resource points to a regular file.

    Returns:
        bool: True if the resource is a file.
    """

  def read_text(self, encoding: str | None = None) -> str:
    """Read the decoded contents of the resource as a string.

    Args:
        encoding (str | None): Text encoding name.

    Returns:
        str: Decoded text content of the resource.
    """


def _resolve_resource_file(framework: str) -> ResourceFileProtocol:
  """Resolve snapshot resource file from ml_ecosystem_snapshots.snapshots.

  Args:
      framework (str): The framework name key.

  Returns:
      ResourceFileProtocol: A traversable resource file path for the framework JSON.
  """
  return importlib.resources.files("ml_ecosystem_snapshots.snapshots").joinpath(f"{framework}.json")


@lru_cache(maxsize=None)
def load_definitions(framework: str) -> dict[str, StandardMap]:
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
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
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
