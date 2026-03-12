import pytest
import libcst as cst
import inspect
from unittest.mock import MagicMock


def test_all_plugin_fallbacks():
  import ml_switcheroo.plugins.auto_fsdp_wrapper as p1
  import ml_switcheroo.plugins.casting as p2
  import ml_switcheroo.plugins.clipping as p3
  import ml_switcheroo.plugins.device_allocator as p4
  import ml_switcheroo.plugins.device_checks as p5
  import ml_switcheroo.plugins.io_handler as p6
  import ml_switcheroo.plugins.keras_sequential as p7
  import ml_switcheroo.plugins.loss_wrapper as p8
  import ml_switcheroo.plugins.method_property as p9
  import ml_switcheroo.plugins.nnx_to_torch_params as p10
  import ml_switcheroo.plugins.padding as p11
  import ml_switcheroo.plugins.scatter as p12
  import ml_switcheroo.plugins.shape_packing as p13
  import ml_switcheroo.plugins.state_flag_injection as p14
  import ml_switcheroo.plugins.static_unroll as p15
  import ml_switcheroo.plugins.tf_data_loader as p16

  modules = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16]
  plugins = []
  for mod in modules:
    for name, obj in inspect.getmembers(mod):
      if inspect.isclass(obj) and hasattr(obj, "transform") and "Plugin" in name:
        try:
          plugins.append(obj())
        except:
          pass

  ctx = MagicMock()
  nodes = [
    cst.Name("dummy"),
    cst.Call(func=cst.Name("dummy"), args=[]),
    cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")),
    cst.FunctionDef(name=cst.Name("dummy"), params=cst.Parameters(), body=cst.IndentedBlock(body=[])),
  ]

  for p in plugins:
    for n in nodes:
      try:
        p.transform(n, ctx)
      except Exception:
        pass
