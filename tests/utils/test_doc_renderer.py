"""Docstring."""

from ml_switcheroo.utils.doc_renderer import OpPageRenderer


def test_render_rst_full():
  """Docstring."""
  renderer = OpPageRenderer()
  context = {
    "name": "Abs",
    "description": "Computes absolute value.",
    "args": ["x: Array", "y"],
    "variants": [
      {
        "framework": "Torch",
        "api": "torch.abs",
        "doc_url": "https://pytorch.org",
        "sharding_supported": True,
        "implementation_type": "Direct API",
        "notes": "Some notes",
      },
      {
        "framework": "JAX",
        "api": "jnp.abs",
        "doc_url": None,
        "sharding_supported": False,
        "implementation_type": "Plugin (fallback)",
        "notes": None,
      },
    ],
  }

  rst = renderer.render_rst(context)
  assert "Abs" in rst
  assert "Computes absolute value." in rst
  assert "x: Array" in rst
  assert "Torch" in rst
  assert "JAX" in rst
  assert "torch.abs" in rst
  assert "jnp.abs" in rst


def test_render_rst_no_variants_no_args():
  """Docstring."""
  renderer = OpPageRenderer()
  context = {"name": "Dummy", "description": "Dummy desc.", "args": [], "variants": []}

  rst = renderer.render_rst(context)
  assert "Dummy" in rst
  assert "Dummy desc." in rst


def test_rst_header_structure():
  """Docstring."""
  renderer = OpPageRenderer()
  context = {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [{"framework": "Torch", "api": "torch.nn.Linear"}],
  }
  rst = renderer.render_rst(context)
  assert "Linear" in rst


def test_rst_args_block():
  """Docstring."""
  renderer = OpPageRenderer()
  context = {"name": "Linear", "description": "Linear transformation.", "args": ["in: int", "out: int"], "variants": []}
  rst = renderer.render_rst(context)
  assert "in: int, out: int" in rst


def test_html_injection():
  """Docstring."""
  renderer = OpPageRenderer()
  context = {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [{"framework": "Torch", "api": "torch.nn.Linear"}],
  }
  rst = renderer.render_rst(context)
  assert "raw:: html" in rst


def test_html_tabs_content():
  """Docstring."""
  renderer = OpPageRenderer()
  variants = [
    {
      "framework": "PyTorch",
      "api": "torch.nn.Linear",
      "implementation_type": "Direct Mapping",
      "doc_url": "http://torch.docs/Linear",
    },
    {"framework": "JAX", "api": "flax.nnx.Linear", "implementation_type": "Direct Mapping", "doc_url": None},
  ]
  html = renderer._render_html_tabs(variants)
  assert '<button class="op-tab-btn active"' in html
  assert ">PyTorch</button>" in html
  assert ">JAX</button>" in html
  assert '<div id="PyTorch_0" class="op-tab-pane active">' in html
  assert '<div id="JAX_1" class="op-tab-pane ">' in html
  assert "torch.nn.Linear" in html
  assert "Direct Mapping" in html
  assert '<a href="http://torch.docs/Linear"' in html
  assert "flax.nnx.Linear" in html
  jax_block_start = html.find('id="JAX_1"')
  jax_block = html[jax_block_start:]
  assert "Official Docs" not in jax_block
