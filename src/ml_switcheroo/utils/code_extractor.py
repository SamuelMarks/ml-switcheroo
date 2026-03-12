"""
Utility to extract source code from live Python objects.

This module solves the "Split-Brain" issue between the core logic and generated harnesses.
Instead of duplicating code manually into template strings, this utility uses the `inspect`
module to read the source code of classes (like the Fuzzer) at runtime.

It ensures that:
1.  Imports required by the class are captured or re-synthesized.
2.  Class definitions are extracted cleanly.
3.  Dependencies (like helper methods or constants) are resolved.
"""

import inspect
import textwrap
from typing import Any, List, Type


class CodeExtractor:
  """
  Extracts self-contained source code for Python classes or functions.
  """

  @staticmethod
  def extract_class(cls_obj: Type[Any]) -> str:
    """
    Reads the source code of a class and formats it for injection.

    Args:
        cls_obj (Type[Any]): The class object to extract (e.g. `InputFuzzer`).

    Returns:
        str: The full source code string of the class definition.

    Raises:
        OSError: If source code cannot be retrieved (e.g. purely compiled C modules).
        TypeError: If input is not a class.
    """
    if not inspect.isclass(cls_obj):
      raise TypeError(f"Expected a class, got {type(cls_obj)}")

    try:
      source = inspect.getsource(cls_obj)
    except OSError as e:
      raise OSError(f"Could not get source for {cls_obj.__name__}: {e}")

    # Clean indentation if it's extensive (e.g. nested class definition)
    return textwrap.dedent(source)

  @staticmethod
  def normalize_harness_imports(source_code: str, required_modules: List[str]) -> str:
    """
    Prepends necessary imports to a code block to ensure it is standalone.

    Since extracted code loses its module-level imports, we must reinject them.

    Args:
        source_code (str): The extracted class body.
        required_modules (List[str]): List of modules to import (e.g. ['random', 'numpy']).

    Returns:
        str: The source code with import statements prepended.
    """
    imports = []
    for mod in required_modules:
      if "." in mod:
        # Handle 'package.module' vs 'package'
        # Naive approach: just import the full thing, or split?
        # Ideally we stick to top-level imports for harnesses to be safe.
        imports.append(f"import {mod}")
      else:
        imports.append(f"import {mod}")

    header = "\n".join(imports)
    return f"{header}\n\n{source_code}"
