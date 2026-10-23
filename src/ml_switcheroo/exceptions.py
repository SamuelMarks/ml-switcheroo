"""Canonical Domain Exception Hierarchy for ML Switcheroo."""


class MLSwitcherooError(Exception):
  """Base exception for all ML Switcheroo domain errors."""


SwitcherooError = MLSwitcherooError


class CSTValidationError(MLSwitcherooError):
  """Raised when a Concrete Syntax Tree (CST) node or structure is invalid."""


class TranslationError(MLSwitcherooError):
  """Raised when AST/CST code translation or transpilation fails."""


class TranspilationError(TranslationError):
  """Alias for TranslationError raised during code transpilation."""


class SnapshotLoadError(MLSwitcherooError):
  """Raised when loading or validating a framework API snapshot fails."""


class SnapshotGroundingError(SnapshotLoadError):
  """Raised when an operator mapping or variant fails snapshot grounding validation."""


class SemanticsResolutionError(MLSwitcherooError):
  """Raised when resolving an abstract operation or framework variant fails."""


class SemanticLoadError(SemanticsResolutionError):
  """Raised when ODL specifications, overlays, or framework configurations fail to load or validate."""


class MacroExpansionError(TranslationError):
  """Raised when expanding a declarative macro template fails."""


class PreambleInjectionError(TranslationError):
  """Raised when parsing or injecting preamble statements into a CST fails."""


class PluginLoadError(MLSwitcherooError):
  """Raised when dynamically loading or registering a plugin hook fails."""


class PluginExecutionError(TranslationError):
  """Raised when a plugin hook encounters an unrecoverable transformation error."""


HookExecutionError = PluginExecutionError


class GraphExtractionError(MLSwitcherooError):
  """Raised when extracting a LogicalGraph IR from source AST fails."""


class TopologicalSortError(GraphExtractionError, ValueError):
  """Raised when a LogicalGraph contains a cycle or invalid topological ordering."""


class DialectLoweringError(TranslationError):
  """Raised when lowering between IR dialects (e.g., MLIR, StableHLO, ISA) fails."""


class BackendEmissionError(DialectLoweringError):
  """Raised when a compiler backend fails to lower or emit target code."""


class SyntaxParseError(MLSwitcherooError, ValueError):
  """Raised when parsing source text into a domain CST fails."""


class IngestionParseError(SyntaxParseError):
  """Raised when source code or intermediate representation fails to parse during ingestion."""


class VerificationExecutionError(MLSwitcherooError):
  """Raised when equivalence verification or fuzzing fails during target execution."""


__all__ = [
  "BackendEmissionError",
  "CSTValidationError",
  "DialectLoweringError",
  "GraphExtractionError",
  "HookExecutionError",
  "IngestionParseError",
  "MLSwitcherooError",
  "MacroExpansionError",
  "PluginExecutionError",
  "PluginLoadError",
  "PreambleInjectionError",
  "SemanticLoadError",
  "SemanticsResolutionError",
  "SnapshotGroundingError",
  "SnapshotLoadError",
  "SwitcherooError",
  "SyntaxParseError",
  "TopologicalSortError",
  "TranslationError",
  "TranspilationError",
  "VerificationExecutionError",
]
