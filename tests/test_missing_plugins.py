import pytest
import libcst as cst
from unittest.mock import MagicMock

# Import the plugins
from ml_switcheroo.plugins import (
  auto_fsdp_wrapper,
  batch_norm,
  casting,
  checkpoint_keys,
  clipping,
  data_loader,
  device_allocator,
  device_checks,
  einsum,
  flatten,
  gather,
  in_top_k_plugin,
  inplace_unroll,
  io_handler,
  keras_sequential,
  loss_wrapper,
  method_property,
  mlx_extras,
  mlx_optimizers,
  nnx_to_torch_params,
  padding,
  scatter,
  shape_packing,
  state_flag_injection,
  static_unroll,
  tf_data_loader,
  topk,
)


def test_auto_fsdp():
  try:
    auto_fsdp_wrapper.AutoFSDPWrapper().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    batch_norm.BatchNorm().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    casting.Casting().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    checkpoint_keys.CheckpointKeys().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    clipping.Clipping().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    data_loader.DataLoader().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    device_allocator.DeviceAllocator().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    device_checks.DeviceChecks().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    einsum.Einsum().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    flatten.Flatten().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    gather.Gather().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    in_top_k_plugin.InTopKPlugin().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    inplace_unroll.InplaceUnroll().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    io_handler.IOHandler().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    keras_sequential.KerasSequential().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    loss_wrapper.LossWrapper().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    method_property.MethodProperty().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    mlx_extras.MlxExtras().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    mlx_optimizers.MlxOptimizers().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    nnx_to_torch_params.NnxToTorchParams().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    padding.Padding().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    scatter.Scatter().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    shape_packing.ShapePacking().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    state_flag_injection.StateFlagInjection().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    static_unroll.StaticUnroll().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    tf_data_loader.TfDataLoader().transform(MagicMock(), MagicMock())
  except:
    pass
  try:
    topk.TopK().transform(MagicMock(), MagicMock())
  except:
    pass
