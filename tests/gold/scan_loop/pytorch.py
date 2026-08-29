"""Test suite for the Pytorch module."""

import typing

import torch


def rnn_loop(cell: typing.Any, x: torch.Tensor, init_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Helper to rnn loop."""
  outputs: list[torch.Tensor] = []
  state = init_state
  for i in range(x.size(0)):
    out, state = cell(x[i], state)
    outputs.append(out)
  return torch.stack(outputs, dim=0), state
