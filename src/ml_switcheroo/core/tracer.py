"""
Transpilation Trace Logger.

This module provides the infrastructure to record the step-by-step execution
of the transpiler. It captures:
1. Lifecycle Phases (Parsing, Analysis, Rewriting).
2. Semantics Matches (Found `torch.abs` -> Map to `Abs` op).
3. AST Mutations (Node A transformed to Node B).

The output is a structured list of Event Log dictionaries suitable for JSON serialization.
"""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class TraceEventType(str, Enum):
  PHASE_START = "phase_start"
  PHASE_END = "phase_end"
  MATCH_SEMANTICS = "match_semantics"
  AST_MUTATION = "ast_mutation"
  ANALYSIS_WARNING = "analysis_warning"
  IMPORT_ACTION = "import_action"
  # New event for debugging decisions
  INSPECTION = "inspection"


@dataclass
class TraceEvent:
  id: str
  type: TraceEventType
  timestamp: float
  description: str
  parent_id: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


class TraceLogger:
  """
  Records transpilation events for visualization.
  Designed to be injected into the Engine and Rewriter.
  """

  def __init__(self):
    self._events: List[TraceEvent] = []
    self._active_phases: List[str] = []  # Stack of phase IDs

  def start_phase(self, name: str, description: str = "") -> str:
    """Starts a nested phase (e.g., 'Rewriting Function X'). Returns Phase ID."""
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

  def end_phase(self):
    """Ends the current active phase."""
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

  def log_match(self, source_api: str, target_api: str, abstract_op: str):
    """Logs a Semantic Match event."""
    self._log_simple(
      TraceEventType.MATCH_SEMANTICS,
      f"Mapped {source_api} -> {target_api}",
      {"source": source_api, "target": target_api, "abstract": abstract_op},
    )

  def log_mutation(self, node_type: str, before: str, after: str):
    """Logs an AST transformation."""
    self._log_simple(TraceEventType.AST_MUTATION, f"Transformed {node_type}", {"before": before, "after": after})

  def log_warning(self, message: str):
    self._log_simple(TraceEventType.ANALYSIS_WARNING, message, {"level": "warning"})

  def log_inspection(self, node_str: str, outcome: str, detail: str = ""):
    """Logs a decision point where no change occurred."""
    self._log_simple(TraceEventType.INSPECTION, f"Inspecting '{node_str}'", {"outcome": outcome, "detail": detail})

  def _log_simple(self, evt_type: TraceEventType, desc: str, meta: Dict[str, Any]):
    parent = self._active_phases[-1] if self._active_phases else None
    self._events.append(
      TraceEvent(
        id=str(uuid.uuid4()), type=evt_type, timestamp=time.time(), description=desc, parent_id=parent, metadata=meta
      )
    )

  def export(self) -> List[Dict[str, Any]]:
    """Returns list of dicts for JSON serialization."""
    return [asdict(e) for e in self._events]


# Global/Contextual instance for ease of access in deep mixins
# In a rigorous implementation, pass this via Engine Config.
_GLOBAL_TRACER = TraceLogger()


def get_tracer() -> TraceLogger:
  return _GLOBAL_TRACER


def reset_tracer():
  global _GLOBAL_TRACER
  _GLOBAL_TRACER = TraceLogger()
