"""
Transpilation Trace Logger.

This module provides the infrastructure to record the step-by-step execution
of the transpiler. It captures:
1. Lifecycle Phases (Parsing, Analysis, Rewriting).
2. Semantics Matches (Found `torch.abs` -> Map to `Abs` op).
3. AST Mutations (Node A transformed to Node B).
4. **Source Line Mapping** for interactive visualization.
5. **AST Snapshots** for visual graph debugging.

The output is a structured list of Event Log dictionaries suitable for JSON serialization.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TraceEventType(str, Enum):
  """
  Enumeration of supported trace event types.
  """

  PHASE_START = "phase_start"
  PHASE_END = "phase_end"
  MATCH_SEMANTICS = "match_semantics"
  AST_MUTATION = "ast_mutation"
  ANALYSIS_WARNING = "analysis_warning"
  IMPORT_ACTION = "import_action"
  INSPECTION = "inspection"
  AST_SNAPSHOT = "ast_snapshot"


@dataclass
class TraceEvent:
  """
  A single unit of execution history.
  """

  id: str
  type: TraceEventType
  timestamp: float
  description: str
  parent_id: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  lineno: Optional[int] = None


class TraceLogger:
  """
  Records transpilation events for visualization.
  Designed to be injected into the Engine and Rewriter.

  This class maintains a stack of active phases to establish parent-child
  relationships between events (e.g. an AST mutation inside a specific Rewrite pass).
  """

  def __init__(self) -> None:
    """Initializes a new TraceLogger."""
    self._events: List[TraceEvent] = []
    self._active_phases: List[str] = []  # Stack of phase IDs

  def start_phase(self, name: str, description: str = "") -> str:
    """
    Starts a nested phase (e.g., 'Rewriting Function X').

    Args:
        name (str): The name of the phase.
        description (str): Additional detail.

    Returns:
        str: The generated Phase ID.
    """
    phase_id = str(uuid.uuid4())

    # determine parent
    parent = self._active_phases[-1] if self._active_phases else None

    event = TraceEvent(
      id=phase_id,
      type=TraceEventType.PHASE_START,
      timestamp=time.time(),
      description=name,
      parent_id=parent,
      metadata={"detail": description},
    )
    self._events.append(event)
    self._active_phases.append(phase_id)
    return phase_id

  def end_phase(self) -> None:
    """
    Ends the current active phase.
    Pops the phase ID from the stack and logs an end event.
    """
    if not self._active_phases:
      return

    phase_id = self._active_phases.pop()
    event = TraceEvent(
      id=str(uuid.uuid4()),
      type=TraceEventType.PHASE_END,
      timestamp=time.time(),
      description="End Phase",
      parent_id=phase_id,
    )
    self._events.append(event)

  def log_match(
    self,
    source_api: str,
    target_api: str,
    abstract_op: str,
    lineno: Optional[int] = None,
  ) -> None:
    """
    Logs a Semantic Match event.

    Args:
        source_api: The source function name (e.g. `torch.abs`).
        target_api: The target implementation (e.g. `jax.numpy.abs`).
        abstract_op: The abstract standard key (e.g. `Abs`).
        lineno: Optional 1-based source line number.
    """
    self._log_simple(
      TraceEventType.MATCH_SEMANTICS,
      f"Mapped {source_api} -> {target_api}",
      {"source": source_api, "target": target_api, "abstract": abstract_op},
      lineno=lineno,
    )

  def log_mutation(self, node_type: str, before: str, after: str, lineno: Optional[int] = None) -> None:
    """
    Logs an AST transformation with code diffs.

    Args:
        node_type: Description of node being changed (e.g. "Call").
        before: Source code snapshot before mutation.
        after: Source code snapshot after mutation.
        lineno: Optional 1-based source line number.
    """
    self._log_simple(
      TraceEventType.AST_MUTATION,
      f"Transformed {node_type}",
      {"before": before, "after": after},
      lineno=lineno,
    )

  def log_warning(self, message: str, lineno: Optional[int] = None) -> None:
    """
    Logs an analysis warning.

    Args:
        message: The warning text.
        lineno: Optional 1-based source line number.
    """
    self._log_simple(
      TraceEventType.ANALYSIS_WARNING,
      message,
      {"level": "warning"},
      lineno=lineno,
    )

  def log_inspection(self, node_str: str, outcome: str, detail: str = "") -> None:
    """
    Logs a decision point where no change occurred (for debugging).

    Args:
        node_str: The code element being inspected.
        outcome: The result (e.g., "Skipped", "Ignored").
        detail: Reason for the outcome.
    """
    self._log_simple(
      TraceEventType.INSPECTION,
      f"Inspecting '{node_str}'",
      {"outcome": outcome, "detail": detail},
    )

  def log_snapshot(self, description: str, mermaid_graph: str, code_snapshot: Optional[str] = None) -> None:
    """
    Logs a full AST snapshot for visualization.

    Args:
        description: Label for the snapshot (e.g. "Before Pivot").
        mermaid_graph: string containing Mermaid diagram definition.
        code_snapshot: Optional string containing source code state.
    """
    self._log_simple(
      TraceEventType.AST_SNAPSHOT,
      description,
      {"mermaid": mermaid_graph, "code": code_snapshot},
    )

  def _log_simple(
    self,
    evt_type: TraceEventType,
    desc: str,
    meta: Dict[str, Any],
    lineno: Optional[int] = None,
  ) -> None:
    """Helper to create and append events."""
    parent = self._active_phases[-1] if self._active_phases else None
    self._events.append(
      TraceEvent(
        id=str(uuid.uuid4()),
        type=evt_type,
        timestamp=time.time(),
        description=desc,
        parent_id=parent,
        metadata=meta,
        lineno=lineno,
      )
    )

  def export(self) -> List[Dict[str, Any]]:
    """
    Exports the event log as a list of dictionaries.

    Returns:
        List[Dict[str, Any]]: JSON-serializable event stream.
    """
    return [asdict(e) for e in self._events]


# Global/Contextual instance for ease of access in deep mixins
_GLOBAL_TRACER = TraceLogger()


def get_tracer() -> TraceLogger:
  """
  Returns the global singleton TraceLogger instance.
  """
  return _GLOBAL_TRACER


def reset_tracer() -> None:
  """
  Resets the global tracer state. Useful between runs or tests.
  """
  global _GLOBAL_TRACER
  _GLOBAL_TRACER = TraceLogger()
