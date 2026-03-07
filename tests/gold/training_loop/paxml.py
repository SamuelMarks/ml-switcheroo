"""Module docstring."""

from praxis import base_layer
from praxis import pytypes


def train_step(task, x, y):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PaxML relies on Tasks and Learners instead of raw custom loops usually
  # FProp, BProp are structurally defined in the trainer context.
  pass
  # </SWITCHEROO_FAILED_TO_TRANS>
