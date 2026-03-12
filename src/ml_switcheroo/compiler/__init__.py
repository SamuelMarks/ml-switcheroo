"""Compiler Package.

This package defines the core Intermediate Representation (IR) and Backend interfaces
for the ml-switcheroo compilation pipeline. It separates the logical definition
of a computation graph from the frontend parsing and backend generation logic.
"""

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, LogicalMesh, PartitionSpec, LogicalAxis
from ml_switcheroo.compiler.sharding import ShardingInferencePass
from ml_switcheroo.compiler.sharding_extractor import ShardingExtractionPass
from ml_switcheroo.compiler.fusion import QKVFusionPass, QKVDefusionPass
from ml_switcheroo.compiler.qwen_fusion import (
  SwiGLUFusionPass,
  SwiGLUDefusionPass,
  VisionPatchEmbeddingFusionPass,
  VisionPatchEmbeddingDefusionPass,
)

__all__ = [
  "CompilerBackend",
  "LogicalGraph",
  "LogicalNode",
  "LogicalEdge",
  "LogicalMesh",
  "PartitionSpec",
  "LogicalAxis",
  "ShardingInferencePass",
  "ShardingExtractionPass",
  "QKVFusionPass",
  "QKVDefusionPass",
  "SwiGLUFusionPass",
  "SwiGLUDefusionPass",
  "VisionPatchEmbeddingFusionPass",
  "VisionPatchEmbeddingDefusionPass",
]
