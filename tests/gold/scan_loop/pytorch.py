"""Test suite for the Pytorch module."""

import torch


def rnn_loop(cell, x: torch.Tensor, init_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Helper to rnn loop."""
  outputs = []
  state = init_state
  for i in range(x.size(0)):
    (out, state) = cell(x[i], state)
    outputs.append(out)
  return (torch.stack(outputs, dim=0), state)
