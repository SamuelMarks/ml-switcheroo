"""Module docstring."""

import torch


def rnn_loop(cell, x: torch.Tensor, init_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PyTorch natively unrolls loops in Python imperative mode
  outputs = []
  state = init_state
  # x shape: (seq_len, batch, dim)
  for i in range(x.size(0)):
    out, state = cell(x[i], state)
    outputs.append(out)
  return torch.stack(outputs, dim=0), state
  # </SWITCHEROO_FAILED_TO_TRANS>
