"""Test suite for the Pytorch module."""

import torch


def rnn_loop(cell: torch.Tensor, x: torch.Tensor, init_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Execute `x.size` to support rnn loop in pytorch tests.

  Args:
      cell: Input `cell` parameter value (`torch.Tensor`).
      x: Primary input tensor or array operand.
      init_state: Input `init_state` parameter value (`torch.Tensor`).

  Returns:
      Resulting `tuple[torch.Tensor, torch.Tensor]` from rnn loop.
  """
  outputs: list[torch.Tensor] = []
  state = init_state
  for i in range(x.size(0)):
    out, state = cell(x[i], state)
    outputs.append(out)
  return torch.stack(outputs, dim=0), state
