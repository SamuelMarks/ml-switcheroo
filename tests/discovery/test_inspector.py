from ml_switcheroo.discovery.inspector import ApiInspector


def test_inspect_math_stdlib():
  """
  Verify we can scan the built-in 'math' module.
  """
  inspector = ApiInspector()
  catalog = inspector.inspect("math")

  # 1. Check for known functions
  assert "math.cos" in catalog, "Failed to find math.cos"
  assert "math.sqrt" in catalog, "Failed to find math.sqrt"

  # 2. Check Signature validity
  cos_sig = catalog["math.cos"]
  # math.cos(x)
  assert "x" in cos_sig["params"] or "val" in cos_sig["params"], "Expected parameter 'x' in math.cos"


def test_inspect_argparse():
  """
  Verify we can scan 'argparse' (structure check).
  """
  inspector = ApiInspector()
  catalog = inspector.inspect("argparse")

  # Check for the ArgumentParser class init or methods
  # Griffe maps methods as Class.method
  assert "argparse.ArgumentParser" in catalog or "argparse.ArgumentParser.__init__" in catalog
