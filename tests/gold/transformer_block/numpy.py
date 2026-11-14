"""NumPy golden reference for a Transformer encoder block."""

import numpy as np
import numpy.typing as npt


def _layer_norm(x: npt.NDArray[np.float64], eps: float = 1e-5) -> npt.NDArray[np.float64]:
  """Normalize ``x`` across its last feature axis.

  Args:
      x: Input floating-point array.
      eps: Numerical stability epsilon.

  Returns:
      Zero-mean unit-variance normalized array.
  """
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return np.asarray((x - mean) / np.sqrt(var + eps), dtype=np.float64)


class TransformerBlock:
  """Pure NumPy Transformer encoder block reference."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
  ) -> None:
    """Initialize attention and feed-forward projection matrices.

    Args:
        embed_dim: Model embedding width.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate (no-op in deterministic NumPy inference).
    """
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    scale = 1.0 / np.sqrt(float(embed_dim))
    self.w_qkv = np.eye(embed_dim, dtype=np.float64) * scale
    self.w_ff1 = np.ones((embed_dim, ff_dim), dtype=np.float64) * scale
    self.w_ff2 = np.ones((ff_dim, embed_dim), dtype=np.float64) * scale

  def __call__(
    self,
    x: npt.NDArray[np.float64],
    attn_mask: npt.NDArray[np.float64] | None = None,
  ) -> npt.NDArray[np.float64]:
    """Compute self-attention and feed-forward block output over ``x``.

    Args:
        x: Input sequence array ``(B, T, D)``.
        attn_mask: Optional additive attention mask ``(T, T)``.

    Returns:
        Encoded sequence array ``(B, T, D)``.
    """
    q = x @ self.w_qkv
    scores = (q @ np.swapaxes(q, -1, -2)) / np.sqrt(float(self.embed_dim // self.num_heads))
    if attn_mask is not None:
      scores = scores + attn_mask
    shifted = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = shifted / np.sum(shifted, axis=-1, keepdims=True)
    attn_out = weights @ x
    h = _layer_norm(x + attn_out)
    ff_hidden = np.maximum(0.0, h @ self.w_ff1)
    return _layer_norm(h + (ff_hidden @ self.w_ff2))
