"""
Discovery Package.

This package implements the Knowledge Acquisition layer of ml-switcheroo.
It is responsible for identifying operations in machine learning libraries,
aligning them with abstract standards, and populating the Semantic Knowledge Base.

Modules:
    - ``inspector``: Low-level introspection of Python modules/objects (Live & Ghost).
    - ``scaffolder``: Heuristic-based scanning to generate initial mappings.
    - ``consensus``: Algorithms to align divergent API names across frameworks.
    - ``harvester``: Extraction of semantic rules from manual test files.
    - ``syncer``: Linking abstract operation definitions to concrete framework APIs.
"""

from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.discovery.syncer import FrameworkSyncer

__all__ = [
  "ApiInspector",
  "ConsensusEngine",
  "FrameworkSyncer",
  "Scaffolder",
  "SemanticHarvester",
]
