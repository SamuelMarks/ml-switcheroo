"""Domain exceptions for ml_switcheroo.core (re-exported from canonical `ml_switcheroo.exceptions`)."""

from ml_switcheroo.exceptions import (
  CSTValidationError,
  DialectLoweringError,
  GraphExtractionError,
  MacroExpansionError,
  MLSwitcherooError,
  PluginLoadError,
  PreambleInjectionError,
  SemanticsResolutionError,
  SnapshotLoadError,
  SyntaxParseError,
  TranslationError,
  TranspilationError,
)

__all__ = [
  "CSTValidationError",
  "DialectLoweringError",
  "GraphExtractionError",
  "MLSwitcherooError",
  "MacroExpansionError",
  "PluginLoadError",
  "PreambleInjectionError",
  "SemanticsResolutionError",
  "SnapshotLoadError",
  "SyntaxParseError",
  "TranslationError",
  "TranspilationError",
]
