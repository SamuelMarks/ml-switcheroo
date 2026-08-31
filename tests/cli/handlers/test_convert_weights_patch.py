"""Docstring for test_convert_weights_patch module."""


def test_weight_script_generator_rules_no_arrow(tmp_path):
  """Docstring."""
  from unittest.mock import MagicMock, patch
  from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator

  source = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out = tmp_path / "script.py"
  semantics = MagicMock()

  def mock_lookup(aid: str):
    """Docstring."""
    if aid == "Conv2d":
      return (
        "Conv2d",
        {
          "variants": {
            "jax": {"api": "jax.numpy.conv", "layout_map": {"weight": "FLATTEN"}},
            "torch": {"api": "torch.nn.Conv2d"},
          }
        },
      )
    return None

  semantics.get_definition.side_effect = mock_lookup

  config = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator = WeightScriptGenerator(semantics, config)

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    source_adapter = MagicMock()
    target_adapter = MagicMock()

    def adapter_side_effect(fw: str):
      """Docstring."""
      if fw == "torch":
        return source_adapter
      return target_adapter

    mock_get_adapter.side_effect = adapter_side_effect
    assert generator.generate(source, out)
