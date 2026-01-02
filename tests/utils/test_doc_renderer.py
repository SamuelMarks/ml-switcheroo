"""
Tests for OpPageRenderer.

Verifies:
1.  RST Header structure.
2.  Signature line formatting.
3.  HTML raw block structure (Buttons, Panes).
4.  Styling classes presence.
"""

import pytest
from ml_switcheroo.utils.doc_renderer import OpPageRenderer


@pytest.fixture
def renderer():
  return OpPageRenderer()


@pytest.fixture
def sample_context():
  return {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [
      {
        "framework": "PyTorch",
        "api": "torch.nn.Linear",
        "implementation_type": "Direct Mapping",
        "doc_url": "http://torch.docs/Linear",
      },
      {"framework": "JAX", "api": "flax.nnx.Linear", "implementation_type": "Direct Mapping", "doc_url": None},
    ],
  }


def test_rst_header_structure(renderer, sample_context):
  """Verify Title and Desc."""
  rst = renderer.render_rst(sample_context)

  assert "Linear\n======" in rst
  assert "Linear transformation." in rst


def test_rst_args_block(renderer, sample_context):
  """Verify signature block."""
  rst = renderer.render_rst(sample_context)

  assert "**Abstract Signature:**" in rst
  assert "``Linear(in: int, out: int)``" in rst


def test_html_injection(renderer, sample_context):
  """Verify raw html directive."""
  rst = renderer.render_rst(sample_context)

  assert ".. raw:: html" in rst
  # Check for indented HTML content
  assert '    <div class="op-tabs-container">' in rst


def test_html_tabs_content(renderer, sample_context):
  """Verify HTML internal structure logic."""
  html = renderer._render_html_tabs(sample_context["variants"])

  # Buttons
  assert '<button class="op-tab-btn active"' in html  # First is active
  assert ">PyTorch</button>" in html
  assert ">JAX</button>" in html

  # Panes
  assert '<div id="PyTorch_0" class="op-tab-pane active">' in html
  assert '<div id="JAX_1" class="op-tab-pane ">' in html

  # Content Details
  assert "torch.nn.Linear" in html
  assert "Direct Mapping" in html

  # Links
  # Torch has link
  assert '<a href="http://torch.docs/Linear"' in html
  # JAX has none
  assert "flax.nnx.Linear" in html
  # Check logic skipped link for JAX
  if "flax.nnx.Linear" in html:
    # Hacky string search in the JAX block
    jax_block_start = html.find('id="JAX_1"')
    jax_block = html[jax_block_start:]
    assert "Official Docs" not in jax_block
