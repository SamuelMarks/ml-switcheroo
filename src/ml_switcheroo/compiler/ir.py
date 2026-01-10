"""
Intermediate Representation (IR).

This module defines the language-agnostic graph data structures used to represent
Deep Learning models after ingestion from source code (e.g. Python/LibCST) or
explicit definition.

It acts as the contract between the Frontend (Ingestion) and the Backend (Synthesis).
"""

from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict, deque


@dataclass
class LogicalNode:
  """
  Represents a computation unit (Layer) in the graph.
  """

  id: str
  """Unique identifier (e.g. 'conv1')."""

  kind: str
  """Operation type (e.g. 'Conv2d', 'Input', 'Output')."""

  metadata: Dict[str, str] = field(default_factory=dict)
  """Dictionary of configuration parameters (e.g. ``kernel_size=3``)."""


@dataclass
class LogicalEdge:
  """
  Represents data flow between two nodes.
  """

  source: str
  """Source node ID."""

  target: str
  """Target node ID."""


@dataclass
class LogicalGraph:
  """
  Language-agnostic representation of the neural network structure.
  """

  name: str = "Model"
  """Name of the graph model/class."""

  nodes: List[LogicalNode] = field(default_factory=list)
  """Ordered list of nodes in the graph."""

  edges: List[LogicalEdge] = field(default_factory=list)
  """List of directed edges between nodes."""


def topological_sort(graph: LogicalGraph) -> List[LogicalNode]:
  """
  Sorts graph nodes by dependency order.

  Ensures that for every edge u -> v, u appears before v in the returned list.
  Handles disconnected components and cycles gracefully by appending
  unreachable nodes in their original definition order.

  Args:
      graph: The logical graph to sort.

  Returns:
      List of nodes in execution order.
  """
  adj = defaultdict(list)
  in_degree = defaultdict(int)
  nodes_by_id = {n.id: n for n in graph.nodes}

  # Initialize in-degree for all nodes
  for n in graph.nodes:
    in_degree[n.id] = 0

  # Build adjacency and degree maps
  for edge in graph.edges:
    adj[edge.source].append(edge.target)
    in_degree[edge.target] += 1

  # Simple queue-based toposort
  # Note: Using sorted keys for determinism in queue initialization
  initial_roots = sorted([n.id for n in graph.nodes if in_degree[n.id] == 0])
  queue = deque(initial_roots)
  sorted_nodes = []

  while queue:
    u = queue.popleft()
    if u in nodes_by_id:
      sorted_nodes.append(nodes_by_id[u])

    for v in adj[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0:
        queue.append(v)

  # Handle disconnected components or cycles by appending remaining nodes
  if len(sorted_nodes) < len(graph.nodes):
    seen = {n.id for n in sorted_nodes}
    # Append remaining nodes in definition order (fallback)
    for n in graph.nodes:
      if n.id not in seen:
        sorted_nodes.append(n)

  return sorted_nodes
