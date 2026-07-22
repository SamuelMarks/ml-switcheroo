"""Test suite for the Doc Renderer module."""

import pytest
from ml_switcheroo.utils.doc_renderer import OpPageRenderer


@pytest.fixture
def renderer():
  """Provides a mock renderer for testing."""
  return OpPageRenderer()


@pytest.fixture
def sample_context():
  """Provides a mock sample context for testing."""
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
  """Verifies the behavior of rst header structure."""
  rst = renderer.render_rst(sample_context)
  assert "Linear\n======" in rst
  assert "Linear transformation." in rst


def test_rst_args_block(renderer, sample_context):
  """Verifies the behavior of rst arguments block."""
  rst = renderer.render_rst(sample_context)
  assert "**Abstract Signature:**" in rst
  assert "``Linear(in: int, out: int)``" in rst


def test_html_injection(renderer, sample_context):
  """Verifies the behavior of HTML injection."""
  rst = renderer.render_rst(sample_context)
  assert ".. raw:: html" in rst
  assert '    <div class="op-tabs-container">' in rst


def test_html_tabs_content(renderer, sample_context):
  """Verifies the behavior of HTML tabs content."""
  html = renderer._render_html_tabs(sample_context["variants"])
  assert '<button class="op-tab-btn active"' in html
  assert ">PyTorch</button>" in html
  assert ">JAX</button>" in html
  assert '<div id="PyTorch_0" class="op-tab-pane active">' in html
  assert '<div id="JAX_1" class="op-tab-pane ">' in html
  assert "torch.nn.Linear" in html
  assert "Direct Mapping" in html
  assert '<a href="http://torch.docs/Linear"' in html
  assert "flax.nnx.Linear" in html
  if "flax.nnx.Linear" in html:
    jax_block_start = html.find('id="JAX_1"')
    jax_block = html[jax_block_start:]
    assert "Official Docs" not in jax_block
