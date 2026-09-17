"""Docstring."""

from typing import Any, Dict, List

from ml_switcheroo.utils.doc_renderer import OpPageRenderer


def test_render_rst_full() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  context: Dict[str, Any] = {
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

  rst: str = renderer.render_rst(context)
  assert "Abs" in rst
  assert "Computes absolute value." in rst
  assert "x: Array" in rst
  assert "Torch" in rst
  assert "JAX" in rst
  assert "torch.abs" in rst
  assert "jnp.abs" in rst


def test_render_rst_no_variants_no_args() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  context: Dict[str, Any] = {"name": "Dummy", "description": "Dummy desc.", "args": [], "variants": []}

  rst: str = renderer.render_rst(context)
  assert "Dummy" in rst
  assert "Dummy desc." in rst


def test_rst_header_structure() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  context: Dict[str, Any] = {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [{"framework": "Torch", "api": "torch.nn.Linear"}],
  }
  rst: str = renderer.render_rst(context)
  assert "Linear" in rst


def test_rst_args_block() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  context: Dict[str, Any] = {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [],
  }
  rst: str = renderer.render_rst(context)
  assert "in: int, out: int" in rst


def test_html_injection() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  context: Dict[str, Any] = {
    "name": "Linear",
    "description": "Linear transformation.",
    "args": ["in: int", "out: int"],
    "variants": [{"framework": "Torch", "api": "torch.nn.Linear"}],
  }
  rst: str = renderer.render_rst(context)
  assert "raw:: html" in rst


def test_html_tabs_content() -> None:
  """Docstring."""
  renderer: OpPageRenderer = OpPageRenderer()
  variants: List[Dict[str, Any]] = [
    {
      "framework": "PyTorch",
      "api": "torch.nn.Linear",
      "implementation_type": "Direct Mapping",
      "doc_url": "http://torch.docs/Linear",
    },
    {"framework": "JAX", "api": "flax.nnx.Linear", "implementation_type": "Direct Mapping", "doc_url": None},
  ]
  html: str = renderer._render_html_tabs(variants)
  assert '<button class="op-tab-btn active"' in html
  assert ">PyTorch</button>" in html
  assert ">JAX</button>" in html
  assert '<div id="PyTorch_0" class="op-tab-pane active">' in html
  assert '<div id="JAX_1" class="op-tab-pane ">' in html
  assert "torch.nn.Linear" in html
  assert "Direct Mapping" in html
  assert '<a href="http://torch.docs/Linear"' in html
  assert "flax.nnx.Linear" in html
  jax_block_start: int = html.find('id="JAX_1"')
  jax_block: str = html[jax_block_start:]
  assert "Official Docs" not in jax_block


def test_sanitize_description_asterisks_and_backticks() -> None:
  """Verifies escaping of asterisks and closing of unmatched backticks in descriptions."""
  renderer: OpPageRenderer = OpPageRenderer()
  assert renderer._sanitize_description("") == ""
  assert (
    renderer._sanitize_description("where(condition, input, other, *, out=None)")
    == r"where(condition, input, other, \*, out=None)"
  )
  assert renderer._sanitize_description("load(f, *, **kwargs)") == r"load(f, \*, \*\*kwargs)"
  assert renderer._sanitize_description("Wraps XLA's `Slice") == "Wraps XLA's `Slice`"
  # Asterisks inside backticks should NOT be escaped
  assert renderer._sanitize_description("Power: :math:`x^y * z`") == "Power: :math:`x^y * z`"
