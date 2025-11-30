import torch
import torch.utils.checkpoint as checkpoint


def explicit_graph_step(x):
  # This standard op SHOULD be converted to JAX
  val = torch.abs(x)

  # This framework-specific utility should trigger the Escape Hatch
  # because it is not mapped in the semantics and requires explicit handling.
  out = checkpoint.checkpoint(lambda v: v * 2, val)

  return out
