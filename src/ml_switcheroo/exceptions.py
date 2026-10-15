"""Canonical Domain Exception Hierarchy for ML Switcheroo."""


class MLSwitcherooError(Exception):
  """Base exception for all ML Switcheroo domain errors."""


class CSTValidationError(MLSwitcherooError):
  """Raised when a Concrete Syntax Tree (CST) node or structure is invalid."""


class TranslationError(MLSwitcherooError):
  """Raised when AST/CST code translation or transpilation fails."""


class TranspilationError(TranslationError):
  """Alias for TranslationError raised during code transpilation."""


class SnapshotLoadError(MLSwitcherooError):
  """Raised when loading or validating a framework API snapshot fails."""


class SemanticsResolutionError(MLSwitcherooError):
  """Raised when resolving an abstract operation or framework variant fails."""


class MacroExpansionError(TranslationError):
  """Raised when expanding a declarative macro template fails."""


class PreambleInjectionError(TranslationError):
  """Raised when parsing or injecting preamble statements into a CST fails."""


class PluginLoadError(MLSwitcherooError):
  """Raised when dynamically loading or registering a plugin hook fails."""


class GraphExtractionError(MLSwitcherooError):
  """Raised when extracting a LogicalGraph IR from source AST fails."""


class DialectLoweringError(TranslationError):
  """Raised when lowering between IR dialects (e.g., MLIR, StableHLO, ISA) fails."""


class SyntaxParseError(MLSwitcherooError):
  """Raised when parsing source text into a domain CST fails."""


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
