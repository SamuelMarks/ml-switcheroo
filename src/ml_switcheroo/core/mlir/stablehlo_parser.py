"""StableHLO Parser module.

This module provides the StableHloParser, which parses StableHLO textual representation.
It leverages the existing MlirParser infrastructure to produce an MLIR CST.
"""

from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.cst import ModuleNode


class StableHloParser:
  """Parses StableHLO textual representation."""

  def __init__(self, text: str):
    """Initialize the StableHLO parser.

    Args:
        text (str): The StableHLO source code to parse.
    """
    self.parser = MlirParser(text)

  def parse(self) -> ModuleNode:
    """Parses the text into an MLIR CST ModuleNode.

    Returns:
        ModuleNode: The root node of the parsed CST.
    """
    return self.parser.parse()
