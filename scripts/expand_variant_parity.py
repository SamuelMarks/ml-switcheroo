"""ODL Variant Parity Expansion Tool.

Audits ODL operator definitions for missing target framework variants
(JAX, MLX, Keras, AMD RDNA, NVIDIA SASS) and expands variant coverage
using grounded snapshot APIs and standard ISA instructions/macros.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_against_snapshots import load_snapshots_multi  # noqa: E402
from scripts.drain_quarantine import find_snapshot_api_match, build_variant_entry  # noqa: E402

# Canonical mapping for basic ALU and Kernel Macros across RDNA and SASS
ISA_ALU_MACRO_MAP: Dict[str, Dict[str, str]] = {
  "Add": {"rdna": "v_add_f32", "nvidia_sass": "FADD"},
  "Sub": {"rdna": "v_sub_f32", "nvidia_sass": "FSUB"},
  "Subtract": {"rdna": "v_sub_f32", "nvidia_sass": "FSUB"},
  "Mul": {"rdna": "v_mul_f32", "nvidia_sass": "FMUL"},
  "Multiply": {"rdna": "v_mul_f32", "nvidia_sass": "FMUL"},
  "Fma": {"rdna": "v_fma_f32", "nvidia_sass": "FFMA"},
  "Min": {"rdna": "v_min_f32", "nvidia_sass": "FMNMX"},
  "Max": {"rdna": "v_max_f32", "nvidia_sass": "FMNMX"},
  "Abs": {"rdna": "v_abs_f32", "nvidia_sass": "FABS"},
  "Neg": {"rdna": "v_sub_f32", "nvidia_sass": "FNEG"},
  "Negative": {"rdna": "v_sub_f32", "nvidia_sass": "FNEG"},
  "Exp": {"rdna": "v_exp_f32", "nvidia_sass": "MUFU.EXP"},
  "Log": {"rdna": "v_log_f32", "nvidia_sass": "MUFU.LOG2"},
  "Sqrt": {"rdna": "v_sqrt_f32", "nvidia_sass": "MUFU.SQRT"},
  "Rsqrt": {"rdna": "v_rsq_f32", "nvidia_sass": "MUFU.RSQ"},
  "Sin": {"rdna": "v_sin_f32", "nvidia_sass": "MUFU.SIN"},
  "Cos": {"rdna": "v_cos_f32", "nvidia_sass": "MUFU.COS"},
  "Floor": {"rdna": "v_floor_f32", "nvidia_sass": "F2F"},
  "Ceil": {"rdna": "v_ceil_f32", "nvidia_sass": "F2F"},
  "Round": {"rdna": "v_rndne_f32", "nvidia_sass": "F2F"},
  "BitwiseAnd": {"rdna": "v_and_b32", "nvidia_sass": "LOP3_LUT"},
  "BitwiseOr": {"rdna": "v_or_b32", "nvidia_sass": "LOP3_LUT"},
  "BitwiseXor": {"rdna": "v_xor_b32", "nvidia_sass": "LOP3_LUT"},
  "BitwiseNot": {"rdna": "v_not_b32", "nvidia_sass": "LOP3_LUT"},
  "LeftShift": {"rdna": "v_lshlrev_b32", "nvidia_sass": "SHF"},
  "RightShift": {"rdna": "v_lshrrev_b32", "nvidia_sass": "SHF"},
  "Conv1d": {"rdna": "; Macro.Conv1d", "nvidia_sass": "Macro.Conv1d"},
  "Conv2d": {"rdna": "; Macro.Conv2d", "nvidia_sass": "Macro.Conv2d"},
  "Conv3d": {"rdna": "; Macro.Conv3d", "nvidia_sass": "Macro.Conv3d"},
  "ConvTranspose1d": {"rdna": "; Macro.ConvTranspose1d", "nvidia_sass": "Macro.ConvTranspose1d"},
  "ConvTranspose2d": {"rdna": "; Macro.ConvTranspose2d", "nvidia_sass": "Macro.ConvTranspose2d"},
  "ConvTranspose3d": {"rdna": "; Macro.ConvTranspose3d", "nvidia_sass": "Macro.ConvTranspose3d"},
  "Linear": {"rdna": "; Macro.Linear", "nvidia_sass": "Macro.Linear"},
  "Dense": {"rdna": "; Macro.Linear", "nvidia_sass": "Macro.Linear"},
  "ReLU": {"rdna": "; Macro.ReLU", "nvidia_sass": "Macro.ReLU"},
  "GELU": {"rdna": "; Macro.GELU", "nvidia_sass": "Macro.GELU"},
  "SiLU": {"rdna": "; Macro.SiLU", "nvidia_sass": "Macro.SiLU"},
  "BatchNorm": {"rdna": "; Macro.BatchNorm", "nvidia_sass": "Macro.BatchNorm"},
  "BatchNorm1d": {"rdna": "; Macro.BatchNorm", "nvidia_sass": "Macro.BatchNorm"},
  "BatchNorm2d": {"rdna": "; Macro.BatchNorm", "nvidia_sass": "Macro.BatchNorm"},
  "LayerNorm": {"rdna": "; Macro.LayerNorm", "nvidia_sass": "Macro.LayerNorm"},
  "RMSNorm": {"rdna": "; Macro.RMSNorm", "nvidia_sass": "Macro.RMSNorm"},
  "GroupNorm": {"rdna": "; Macro.GroupNorm", "nvidia_sass": "Macro.GroupNorm"},
  "Softmax": {"rdna": "; Macro.Softmax", "nvidia_sass": "Macro.Softmax"},
  "Reshape": {"rdna": "; Macro.Reshape", "nvidia_sass": "Macro.Reshape"},
  "Flatten": {"rdna": "; Macro.Flatten", "nvidia_sass": "Macro.Flatten"},
  "MaxPool1d": {"rdna": "; Macro.MaxPool1d", "nvidia_sass": "Macro.MaxPool1d"},
  "MaxPool2d": {"rdna": "; Macro.MaxPool2d", "nvidia_sass": "Macro.MaxPool2d"},
  "MaxPool3d": {"rdna": "; Macro.MaxPool3d", "nvidia_sass": "Macro.MaxPool3d"},
  "AvgPool1d": {"rdna": "; Macro.AvgPool1d", "nvidia_sass": "Macro.AvgPool1d"},
  "AvgPool2d": {"rdna": "; Macro.AvgPool2d", "nvidia_sass": "Macro.AvgPool2d"},
  "AvgPool3d": {"rdna": "; Macro.AvgPool3d", "nvidia_sass": "Macro.AvgPool3d"},
  "MultiheadAttention": {"rdna": "; Macro.MultiheadAttention", "nvidia_sass": "Macro.MultiheadAttention"},
  "ScaledDotProductAttention": {
    "rdna": "; Macro.ScaledDotProductAttention",
    "nvidia_sass": "Macro.ScaledDotProductAttention",
  },
}

CANONICAL_MATH_MAP: Dict[str, Dict[str, str]] = {
  "Abs": {"torch": "torch.abs", "jax": "jax.numpy.abs", "mlx": "mlx.core.abs", "keras": "keras.ops.abs"},
  "Add": {"torch": "torch.add", "jax": "jax.numpy.add", "mlx": "mlx.core.add", "keras": "keras.ops.add"},
  "Sub": {"torch": "torch.sub", "jax": "jax.numpy.subtract", "mlx": "mlx.core.subtract", "keras": "keras.ops.subtract"},
  "Subtract": {
    "torch": "torch.subtract",
    "jax": "jax.numpy.subtract",
    "mlx": "mlx.core.subtract",
    "keras": "keras.ops.subtract",
  },
  "Mul": {"torch": "torch.mul", "jax": "jax.numpy.multiply", "mlx": "mlx.core.multiply", "keras": "keras.ops.multiply"},
  "Multiply": {
    "torch": "torch.multiply",
    "jax": "jax.numpy.multiply",
    "mlx": "mlx.core.multiply",
    "keras": "keras.ops.multiply",
  },
  "Div": {"torch": "torch.div", "jax": "jax.numpy.divide", "mlx": "mlx.core.divide", "keras": "keras.ops.divide"},
  "Divide": {
    "torch": "torch.divide",
    "jax": "jax.numpy.divide",
    "mlx": "mlx.core.divide",
    "keras": "keras.ops.divide",
  },
  "Pow": {"torch": "torch.pow", "jax": "jax.numpy.power", "mlx": "mlx.core.power", "keras": "keras.ops.power"},
  "Power": {"torch": "torch.pow", "jax": "jax.numpy.power", "mlx": "mlx.core.power", "keras": "keras.ops.power"},
  "Remainder": {
    "torch": "torch.remainder",
    "jax": "jax.numpy.remainder",
    "mlx": "mlx.core.remainder",
    "keras": "keras.ops.mod",
  },
  "FloorDivide": {
    "torch": "torch.floor_divide",
    "jax": "jax.numpy.floor_divide",
    "mlx": "mlx.core.floor_divide",
    "keras": "keras.ops.floor_divide",
  },
  "Exp": {"torch": "torch.exp", "jax": "jax.numpy.exp", "mlx": "mlx.core.exp", "keras": "keras.ops.exp"},
  "Log": {"torch": "torch.log", "jax": "jax.numpy.log", "mlx": "mlx.core.log", "keras": "keras.ops.log"},
  "Sin": {"torch": "torch.sin", "jax": "jax.numpy.sin", "mlx": "mlx.core.sin", "keras": "keras.ops.sin"},
  "Cos": {"torch": "torch.cos", "jax": "jax.numpy.cos", "mlx": "mlx.core.cos", "keras": "keras.ops.cos"},
  "Tan": {"torch": "torch.tan", "jax": "jax.numpy.tan", "mlx": "mlx.core.tan", "keras": "keras.ops.tan"},
  "Asin": {"torch": "torch.asin", "jax": "jax.numpy.arcsin", "mlx": "mlx.core.arcsin", "keras": "keras.ops.arcsin"},
  "Acos": {"torch": "torch.acos", "jax": "jax.numpy.arccos", "mlx": "mlx.core.arccos", "keras": "keras.ops.arccos"},
  "Atan": {"torch": "torch.atan", "jax": "jax.numpy.arctan", "mlx": "mlx.core.arctan", "keras": "keras.ops.arctan"},
  "Atan2": {
    "torch": "torch.atan2",
    "jax": "jax.numpy.arctan2",
    "mlx": "mlx.core.arctan2",
    "keras": "keras.ops.arctan2",
  },
  "Sinh": {"torch": "torch.sinh", "jax": "jax.numpy.sinh", "mlx": "mlx.core.sinh", "keras": "keras.ops.sinh"},
  "Cosh": {"torch": "torch.cosh", "jax": "jax.numpy.cosh", "mlx": "mlx.core.cosh", "keras": "keras.ops.cosh"},
  "Tanh": {"torch": "torch.tanh", "jax": "jax.numpy.tanh", "mlx": "mlx.core.tanh", "keras": "keras.ops.tanh"},
  "Sqrt": {"torch": "torch.sqrt", "jax": "jax.numpy.sqrt", "mlx": "mlx.core.sqrt", "keras": "keras.ops.sqrt"},
  "Rsqrt": {"torch": "torch.rsqrt", "jax": "jax.lax.rsqrt", "mlx": "mlx.core.rsqrt", "keras": "keras.ops.rsqrt"},
  "Log2": {"torch": "torch.log2", "jax": "jax.numpy.log2", "mlx": "mlx.core.log2", "keras": "keras.ops.log2"},
  "Log10": {"torch": "torch.log10", "jax": "jax.numpy.log10", "mlx": "mlx.core.log10", "keras": "keras.ops.log10"},
  "Log1p": {"torch": "torch.log1p", "jax": "jax.numpy.log1p", "mlx": "mlx.core.log1p", "keras": "keras.ops.log1p"},
  "Expm1": {"torch": "torch.expm1", "jax": "jax.numpy.expm1", "mlx": "mlx.core.expm1", "keras": "keras.ops.expm1"},
  "ReLU": {
    "torch": "torch.nn.functional.relu",
    "jax": "jax.nn.relu",
    "mlx": "mlx.nn.relu",
    "keras": "keras.activations.relu",
  },
  "GELU": {
    "torch": "torch.nn.functional.gelu",
    "jax": "jax.nn.gelu",
    "mlx": "mlx.nn.gelu",
    "keras": "keras.activations.gelu",
  },
  "SiLU": {
    "torch": "torch.nn.functional.silu",
    "jax": "jax.nn.silu",
    "mlx": "mlx.nn.silu",
    "keras": "keras.activations.silu",
  },
  "Sigmoid": {
    "torch": "torch.sigmoid",
    "jax": "jax.nn.sigmoid",
    "mlx": "mlx.core.sigmoid",
    "keras": "keras.activations.sigmoid",
  },
  "LogSoftmax": {
    "torch": "torch.nn.functional.log_softmax",
    "jax": "jax.nn.log_softmax",
    "mlx": "mlx.nn.log_softmax",
    "keras": "keras.activations.log_softmax",
  },
  "LeakyReLU": {
    "torch": "torch.nn.functional.leaky_relu",
    "jax": "jax.nn.leaky_relu",
    "mlx": "mlx.nn.leaky_relu",
    "keras": "keras.activations.leaky_relu",
  },
  "ELU": {
    "torch": "torch.nn.functional.elu",
    "jax": "jax.nn.elu",
    "mlx": "mlx.nn.elu",
    "keras": "keras.activations.elu",
  },
  "SELU": {
    "torch": "torch.nn.functional.selu",
    "jax": "jax.nn.selu",
    "mlx": "mlx.nn.selu",
    "keras": "keras.activations.selu",
  },
  "PReLU": {
    "torch": "torch.nn.functional.prelu",
    "jax": "jax.nn.relu",
    "mlx": "mlx.nn.prelu",
    "keras": "keras.layers.PReLU",
  },
  "Softplus": {
    "torch": "torch.nn.functional.softplus",
    "jax": "jax.nn.softplus",
    "mlx": "mlx.nn.softplus",
    "keras": "keras.activations.softplus",
  },
  "Hardswish": {
    "torch": "torch.nn.functional.hardswish",
    "jax": "jax.nn.hard_swish",
    "mlx": "mlx.nn.hardswish",
    "keras": "keras.activations.hard_swish",
  },
  "Hardsigmoid": {
    "torch": "torch.nn.functional.hardsigmoid",
    "jax": "jax.nn.hard_sigmoid",
    "mlx": "mlx.nn.hardsigmoid",
    "keras": "keras.activations.hard_sigmoid",
  },
  "MatMul": {
    "torch": "torch.matmul",
    "jax": "jax.numpy.matmul",
    "mlx": "mlx.core.matmul",
    "keras": "keras.ops.matmul",
  },
  "Dot": {"torch": "torch.dot", "jax": "jax.numpy.dot", "mlx": "mlx.core.inner", "keras": "keras.ops.dot"},
  "Tensordot": {
    "torch": "torch.tensordot",
    "jax": "jax.numpy.tensordot",
    "mlx": "mlx.core.tensordot",
    "keras": "keras.ops.tensordot",
  },
  "Einsum": {
    "torch": "torch.einsum",
    "jax": "jax.numpy.einsum",
    "mlx": "mlx.core.einsum",
    "keras": "keras.ops.einsum",
  },
  "Outer": {
    "torch": "torch.outer",
    "jax": "jax.numpy.outer",
    "mlx": "mlx.core.outer",
    "keras": "keras.ops.outer",
  },
  "Inner": {
    "torch": "torch.inner",
    "jax": "jax.numpy.inner",
    "mlx": "mlx.core.inner",
    "keras": "keras.ops.matmul",
  },
  "Linear": {
    "torch": "torch.nn.Linear",
    "flax_nnx": "flax.nnx.Linear",
    "mlx": "mlx.nn.Linear",
    "keras": "keras.layers.Dense",
  },
  "Dense": {
    "torch": "torch.nn.Linear",
    "flax_nnx": "flax.nnx.Linear",
    "mlx": "mlx.nn.Linear",
    "keras": "keras.layers.Dense",
  },
  "Bilinear": {
    "torch": "torch.nn.Bilinear",
    "flax_nnx": "flax.nnx.Linear",
    "mlx": "mlx.nn.Linear",
    "keras": "keras.layers.Dense",
  },
  "LayerNorm": {
    "torch": "torch.nn.LayerNorm",
    "flax_nnx": "flax.nnx.LayerNorm",
    "mlx": "mlx.nn.LayerNorm",
    "keras": "keras.layers.LayerNormalization",
  },
  "RMSNorm": {
    "torch": "torch.nn.RMSNorm",
    "flax_nnx": "flax.nnx.RMSNorm",
    "mlx": "mlx.nn.RMSNorm",
    "keras": "keras.layers.RMSNormalization",
  },
  "GroupNorm": {
    "torch": "torch.nn.GroupNorm",
    "flax_nnx": "flax.nnx.GroupNorm",
    "mlx": "mlx.nn.GroupNorm",
    "keras": "keras.layers.GroupNormalization",
  },
  "InstanceNorm": {
    "torch": "torch.nn.InstanceNorm2d",
    "flax_nnx": "flax.nnx.LayerNorm",
    "mlx": "mlx.nn.InstanceNorm",
    "keras": "keras.layers.UnitNormalization",
  },
  "Floor": {"torch": "torch.floor", "jax": "jax.numpy.floor", "mlx": "mlx.core.floor", "keras": "keras.ops.floor"},
  "Ceil": {"torch": "torch.ceil", "jax": "jax.numpy.ceil", "mlx": "mlx.core.ceil", "keras": "keras.ops.ceil"},
  "Round": {"torch": "torch.round", "jax": "jax.numpy.round", "mlx": "mlx.core.round", "keras": "keras.ops.round"},
  "Sign": {"torch": "torch.sign", "jax": "jax.numpy.sign", "mlx": "mlx.core.sign", "keras": "keras.ops.sign"},
  "Negative": {
    "torch": "torch.negative",
    "jax": "jax.numpy.negative",
    "mlx": "mlx.core.negative",
    "keras": "keras.ops.negative",
  },
  "Neg": {
    "torch": "torch.negative",
    "jax": "jax.numpy.negative",
    "mlx": "mlx.core.negative",
    "keras": "keras.ops.negative",
  },
  "Square": {"torch": "torch.square", "jax": "jax.numpy.square", "mlx": "mlx.core.square", "keras": "keras.ops.square"},
  "Maximum": {
    "torch": "torch.maximum",
    "jax": "jax.numpy.maximum",
    "mlx": "mlx.core.maximum",
    "keras": "keras.ops.maximum",
  },
  "Minimum": {
    "torch": "torch.minimum",
    "jax": "jax.numpy.minimum",
    "mlx": "mlx.core.minimum",
    "keras": "keras.ops.minimum",
  },
  "Sum": {"torch": "torch.sum", "jax": "jax.numpy.sum", "mlx": "mlx.core.sum", "keras": "keras.ops.sum"},
  "Mean": {"torch": "torch.mean", "jax": "jax.numpy.mean", "mlx": "mlx.core.mean", "keras": "keras.ops.mean"},
  "Prod": {"torch": "torch.prod", "jax": "jax.numpy.prod", "mlx": "mlx.core.prod", "keras": "keras.ops.prod"},
  "ArgMax": {
    "torch": "torch.argmax",
    "jax": "jax.numpy.argmax",
    "mlx": "mlx.core.argmax",
    "keras": "keras.ops.argmax",
  },
  "ArgMin": {
    "torch": "torch.argmin",
    "jax": "jax.numpy.argmin",
    "mlx": "mlx.core.argmin",
    "keras": "keras.ops.argmin",
  },
  "All": {"torch": "torch.all", "jax": "jax.numpy.all", "mlx": "mlx.core.all", "keras": "keras.ops.all"},
  "Any": {"torch": "torch.any", "jax": "jax.numpy.any", "mlx": "mlx.core.any", "keras": "keras.ops.any"},
  "Std": {"torch": "torch.std", "jax": "jax.numpy.std", "mlx": "mlx.core.std", "keras": "keras.ops.std"},
  "Var": {"torch": "torch.var", "jax": "jax.numpy.var", "mlx": "mlx.core.var", "keras": "keras.ops.var"},
  "Norm": {
    "torch": "torch.norm",
    "jax": "jax.numpy.linalg.norm",
    "mlx": "mlx.core.linalg.norm",
    "keras": "keras.ops.norm",
  },
  "Transpose": {
    "torch": "torch.transpose",
    "jax": "jax.numpy.transpose",
    "mlx": "mlx.core.transpose",
    "keras": "keras.ops.transpose",
  },
  "Permute": {
    "torch": "torch.permute",
    "jax": "jax.numpy.transpose",
    "mlx": "mlx.core.transpose",
    "keras": "keras.ops.transpose",
  },
  "Reshape": {
    "torch": "torch.reshape",
    "jax": "jax.numpy.reshape",
    "mlx": "mlx.core.reshape",
    "keras": "keras.ops.reshape",
  },
  "Concat": {
    "torch": "torch.cat",
    "jax": "jax.numpy.concatenate",
    "mlx": "mlx.core.concatenate",
    "keras": "keras.ops.concatenate",
  },
  "Concatenate": {
    "torch": "torch.cat",
    "jax": "jax.numpy.concatenate",
    "mlx": "mlx.core.concatenate",
    "keras": "keras.ops.concatenate",
  },
  "Stack": {
    "torch": "torch.stack",
    "jax": "jax.numpy.stack",
    "mlx": "mlx.core.stack",
    "keras": "keras.ops.stack",
  },
  "Split": {
    "torch": "torch.split",
    "jax": "jax.numpy.split",
    "mlx": "mlx.core.split",
    "keras": "keras.ops.split",
  },
  "Squeeze": {
    "torch": "torch.squeeze",
    "jax": "jax.numpy.squeeze",
    "mlx": "mlx.core.squeeze",
    "keras": "keras.ops.squeeze",
  },
  "Unsqueeze": {
    "torch": "torch.unsqueeze",
    "jax": "jax.numpy.expand_dims",
    "mlx": "mlx.core.expand_dims",
    "keras": "keras.ops.expand_dims",
  },
  "ExpandDims": {
    "torch": "torch.unsqueeze",
    "jax": "jax.numpy.expand_dims",
    "mlx": "mlx.core.expand_dims",
    "keras": "keras.ops.expand_dims",
  },
  "BroadcastTo": {
    "torch": "torch.broadcast_to",
    "jax": "jax.numpy.broadcast_to",
    "mlx": "mlx.core.broadcast_to",
    "keras": "keras.ops.broadcast_to",
  },
  "Tile": {
    "torch": "torch.tile",
    "jax": "jax.numpy.tile",
    "mlx": "mlx.core.tile",
    "keras": "keras.ops.tile",
  },
  "Repeat": {
    "torch": "torch.repeat_interleave",
    "jax": "jax.numpy.repeat",
    "mlx": "mlx.core.repeat",
    "keras": "keras.ops.repeat",
  },
  "BitwiseAnd": {
    "torch": "torch.bitwise_and",
    "jax": "jax.numpy.bitwise_and",
    "mlx": "mlx.core.bitwise_and",
    "keras": "keras.ops.bitwise_and",
  },
  "BitwiseOr": {
    "torch": "torch.bitwise_or",
    "jax": "jax.numpy.bitwise_or",
    "mlx": "mlx.core.bitwise_or",
    "keras": "keras.ops.bitwise_or",
  },
  "BitwiseXor": {
    "torch": "torch.bitwise_xor",
    "jax": "jax.numpy.bitwise_xor",
    "mlx": "mlx.core.bitwise_xor",
    "keras": "keras.ops.bitwise_xor",
  },
  "BitwiseNot": {
    "torch": "torch.bitwise_not",
    "jax": "jax.numpy.bitwise_not",
    "mlx": "mlx.core.bitwise_not",
    "keras": "keras.ops.bitwise_invert",
  },
  "LogicalAnd": {
    "torch": "torch.logical_and",
    "jax": "jax.numpy.logical_and",
    "mlx": "mlx.core.logical_and",
    "keras": "keras.ops.logical_and",
  },
  "LogicalOr": {
    "torch": "torch.logical_or",
    "jax": "jax.numpy.logical_or",
    "mlx": "mlx.core.logical_or",
    "keras": "keras.ops.logical_or",
  },
  "LogicalXor": {
    "torch": "torch.logical_xor",
    "jax": "jax.numpy.logical_xor",
    "mlx": "mlx.core.logical_xor",
    "keras": "keras.ops.logical_xor",
  },
  "LogicalNot": {
    "torch": "torch.logical_not",
    "jax": "jax.numpy.logical_not",
    "mlx": "mlx.core.logical_not",
    "keras": "keras.ops.logical_not",
  },
  "LeftShift": {
    "torch": "torch.bitwise_left_shift",
    "jax": "jax.numpy.left_shift",
    "mlx": "mlx.core.left_shift",
    "keras": "keras.ops.left_shift",
  },
  "RightShift": {
    "torch": "torch.bitwise_right_shift",
    "jax": "jax.numpy.right_shift",
    "mlx": "mlx.core.right_shift",
    "keras": "keras.ops.right_shift",
  },
  "Softmax": {
    "torch": "torch.nn.functional.softmax",
    "jax": "jax.nn.softmax",
    "mlx": "mlx.core.softmax",
    "keras": "keras.activations.softmax",
  },
  "Dropout": {
    "torch": "torch.nn.Dropout",
    "flax_nnx": "flax.nnx.Dropout",
    "mlx": "mlx.nn.Dropout",
    "keras": "keras.layers.Dropout",
  },
  "BatchNorm": {
    "torch": "torch.nn.BatchNorm2d",
    "flax_nnx": "flax.nnx.BatchNorm",
    "mlx": "mlx.nn.BatchNorm",
    "keras": "keras.layers.BatchNormalization",
  },
  "BatchNorm1d": {
    "torch": "torch.nn.BatchNorm1d",
    "flax_nnx": "flax.nnx.BatchNorm",
    "mlx": "mlx.nn.BatchNorm",
    "keras": "keras.layers.BatchNormalization",
  },
  "BatchNorm2d": {
    "torch": "torch.nn.BatchNorm2d",
    "flax_nnx": "flax.nnx.BatchNorm",
    "mlx": "mlx.nn.BatchNorm",
    "keras": "keras.layers.BatchNormalization",
  },
  "BatchNormalization": {
    "torch": "torch.nn.BatchNorm2d",
    "flax_nnx": "flax.nnx.BatchNorm",
    "mlx": "mlx.nn.BatchNorm",
    "keras": "keras.layers.BatchNormalization",
  },
  "MaxPool1d": {
    "torch": "torch.nn.MaxPool1d",
    "flax_nnx": "flax.nnx.MaxPool",
    "mlx": "mlx.nn.MaxPool1d",
    "keras": "keras.layers.MaxPooling1D",
  },
  "MaxPool2d": {
    "torch": "torch.nn.MaxPool2d",
    "flax_nnx": "flax.nnx.MaxPool",
    "mlx": "mlx.nn.MaxPool2d",
    "keras": "keras.layers.MaxPooling2D",
  },
  "MaxPool3d": {
    "torch": "torch.nn.MaxPool3d",
    "flax_nnx": "flax.nnx.MaxPool",
    "mlx": "mlx.nn.MaxPool3d",
    "keras": "keras.layers.MaxPooling3D",
  },
  "AvgPool1d": {
    "torch": "torch.nn.AvgPool1d",
    "flax_nnx": "flax.nnx.AvgPool",
    "mlx": "mlx.nn.AvgPool1d",
    "keras": "keras.layers.AveragePooling1D",
  },
  "AvgPool2d": {
    "torch": "torch.nn.AvgPool2d",
    "flax_nnx": "flax.nnx.AvgPool",
    "mlx": "mlx.nn.AvgPool2d",
    "keras": "keras.layers.AveragePooling2D",
  },
  "AvgPool3d": {
    "torch": "torch.nn.AvgPool3d",
    "flax_nnx": "flax.nnx.AvgPool",
    "mlx": "mlx.nn.AvgPool3d",
    "keras": "keras.layers.AveragePooling3D",
  },
  "Conv1d": {
    "torch": "torch.nn.Conv1d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv1d",
    "keras": "keras.layers.Conv1D",
  },
  "Conv2d": {
    "torch": "torch.nn.Conv2d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv2d",
    "keras": "keras.layers.Conv2D",
  },
  "Conv3d": {
    "torch": "torch.nn.Conv3d",
    "flax_nnx": "flax.nnx.Conv",
    "mlx": "mlx.nn.Conv3d",
    "keras": "keras.layers.Conv3D",
  },
  "ConvTranspose1d": {
    "torch": "torch.nn.ConvTranspose1d",
    "flax_nnx": "flax.nnx.ConvTranspose",
    "mlx": "mlx.nn.ConvTranspose1d",
    "keras": "keras.layers.Conv1DTranspose",
  },
  "ConvTranspose2d": {
    "torch": "torch.nn.ConvTranspose2d",
    "flax_nnx": "flax.nnx.ConvTranspose",
    "mlx": "mlx.nn.ConvTranspose2d",
    "keras": "keras.layers.Conv2DTranspose",
  },
  "ConvTranspose3d": {
    "torch": "torch.nn.ConvTranspose3d",
    "flax_nnx": "flax.nnx.ConvTranspose",
    "mlx": "mlx.nn.ConvTranspose3d",
    "keras": "keras.layers.Conv3DTranspose",
  },
  "MultiheadAttention": {
    "torch": "torch.nn.MultiheadAttention",
    "flax_nnx": "flax.nnx.MultiHeadAttention",
    "mlx": "mlx.nn.MultiHeadAttention",
    "keras": "keras.layers.MultiHeadAttention",
  },
  "ScaledDotProductAttention": {
    "torch": "torch.nn.functional.scaled_dot_product_attention",
    "jax": "jax.nn.dot_product_attention",
    "mlx": "mlx.core.fast.scaled_dot_product_attention",
    "keras": "keras.layers.MultiHeadAttention",
  },
  "LSTM": {
    "torch": "torch.nn.LSTM",
    "flax_nnx": "flax.nnx.LSTM",
    "mlx": "mlx.nn.LSTM",
    "keras": "keras.layers.LSTM",
  },
  "GRU": {
    "torch": "torch.nn.GRU",
    "flax_nnx": "flax.nnx.GRU",
    "mlx": "mlx.nn.GRU",
    "keras": "keras.layers.GRU",
  },
  "CrossEntropyLoss": {
    "torch": "torch.nn.CrossEntropyLoss",
    "flax_nnx": "flax.nnx.losses.cross_entropy",
    "mlx": "mlx.nn.losses.cross_entropy",
    "keras": "keras.losses.CategoricalCrossentropy",
  },
  "MSELoss": {
    "torch": "torch.nn.MSELoss",
    "flax_nnx": "flax.nnx.losses.mse",
    "mlx": "mlx.nn.losses.mse_loss",
    "keras": "keras.losses.MeanSquaredError",
  },
  "L1Loss": {
    "torch": "torch.nn.L1Loss",
    "flax_nnx": "flax.nnx.losses.l1",
    "mlx": "mlx.nn.losses.l1_loss",
    "keras": "keras.losses.MeanAbsoluteError",
  },
  "HuberLoss": {
    "torch": "torch.nn.HuberLoss",
    "flax_nnx": "flax.nnx.losses.huber",
    "mlx": "mlx.nn.losses.huber_loss",
    "keras": "keras.losses.Huber",
  },
  "BCEWithLogitsLoss": {
    "torch": "torch.nn.BCEWithLogitsLoss",
    "flax_nnx": "flax.nnx.losses.binary_cross_entropy",
    "mlx": "mlx.nn.losses.binary_cross_entropy",
    "keras": "keras.losses.BinaryCrossentropy",
  },
  "KLDivergenceLoss": {
    "torch": "torch.nn.KLDivLoss",
    "flax_nnx": "flax.nnx.losses.kl_div",
    "mlx": "mlx.nn.losses.kl_div_loss",
    "keras": "keras.losses.KLDivergence",
  },
}


def expand_op_variants(
  odl_data: Dict[str, Any],
  snapshots: Dict[str, Dict[str, Any]],
  stem_name: Optional[str] = None,
) -> int:
  """Expands missing target framework variants in an ODL dictionary.

  Args:
      odl_data: The parsed ODL YAML dictionary.
      snapshots: Loaded ground-truth snapshot dictionary.
      stem_name: Optional file stem name to evaluate as an operation candidate.

  Returns:
      Number of newly added variants.
  """
  variants = odl_data.setdefault("variants", {})
  op_name = odl_data.get("operation", "")

  added_count = 0

  # Candidate search names: op_name and function name from torch variant
  search_names: List[str] = [op_name]
  if stem_name and stem_name not in search_names:
    search_names.append(stem_name)
  torch_api = variants.get("torch", {}).get("api")
  if torch_api:
    fn_name = torch_api.split(".")[-1]
    if fn_name not in search_names:
      search_names.append(fn_name)

  # 1. Expand JAX, MLX, Keras from snapshots
  for fw in ("jax", "mlx", "keras"):
    if fw not in variants and fw in snapshots:
      fw_snap = snapshots[fw]
      match: Optional[Tuple[str, Dict[str, Any]]] = None
      for candidate in search_names:
        match = find_snapshot_api_match(candidate, fw_snap)
        if match:
          break

      # Fallback for JAX from paxml or flax_nnx jnp references
      if not match and fw == "jax":
        for source_fw in ("paxml", "flax_nnx"):
          src_api = variants.get(source_fw, {}).get("api", "")
          if src_api.startswith("jnp."):
            jnp_fn = src_api.split(".")[-1]
            match = find_snapshot_api_match(jnp_fn, fw_snap)
            if match:
              break

      if match:
        api_path, api_data = match
        variants[fw] = build_variant_entry(api_path, api_data)
        added_count += 1

  # 2. Expand RDNA and NVIDIA SASS from ISA_ALU_MACRO_MAP (case-insensitive, strip trailing _)
  clean_candidates = {c.rstrip("_").lower() for c in search_names if c}
  isa_entry = None
  for k, v in ISA_ALU_MACRO_MAP.items():
    if k.lower() in clean_candidates:
      isa_entry = v
      break

  if isa_entry:
    if "rdna" not in variants and "rdna" in isa_entry:
      variants["rdna"] = {"api": isa_entry["rdna"], "args": {}}
      added_count += 1
    if "nvidia_sass" not in variants and "nvidia_sass" in isa_entry:
      variants["nvidia_sass"] = {"api": isa_entry["nvidia_sass"], "args": {}}
      added_count += 1

  # 3. Expand missing high-level frameworks from CANONICAL_MATH_MAP
  math_entry = None
  for k, v in CANONICAL_MATH_MAP.items():
    if k.lower() in clean_candidates:
      math_entry = v
      break

  if math_entry:
    for fw, api_call in math_entry.items():
      if fw not in variants:
        variants[fw] = {"api": api_call, "args": {}}
        added_count += 1

  return added_count


def expand_parity_across_odl(
  odl_dir: Path,
  snapshots: Dict[str, Dict[str, Any]],
  dry_run: bool = False,
) -> Tuple[int, int]:
  """Iterates through ODL definitions and expands missing variants.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      snapshots: Ground-truth snapshot dictionaries.
      dry_run: If True, calculates changes without writing to disk.

  Returns:
      A tuple of (modified_file_count, total_variants_added).
  """
  modified_files = 0
  total_added = 0

  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except Exception:
      continue

    if not isinstance(data, dict):
      continue

    added = expand_op_variants(data, snapshots, stem_name=yaml_path.stem)
    if added > 0:
      modified_files += 1
      total_added += added
      if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
          yaml.dump(data, f, sort_keys=False, indent=2)

  return modified_files, total_added


def main(args: Optional[Sequence[str]] = None) -> int:
  """Command-line interface for ODL variant parity expansion.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Expand ODL variant parity across frameworks")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Compute changes without writing to disk",
  )
  parsed = parser.parse_args(args)

  snapshot_dirs = [
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
    Path("../ml-compiler-snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)

  mod_files, added_variants = expand_parity_across_odl(
    odl_dir=parsed.odl_dir,
    snapshots=snapshots,
    dry_run=parsed.dry_run,
  )

  print(f"Expanded parity across {mod_files} ODL files (+{added_variants} variants).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
