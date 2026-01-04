"""
Template for the Verification Harness.

This template is populated by `HarnessGenerator`. It provides a completely generic
skeleton for running `source_func(x) == target_func(x)` checks.

Key placeholders:
- {imports}: Framework-specific imports (e.g., 'import jax').
- {init_helpers}: Framework-specific helper function definitions (e.g., '_make_jax_key').
- {param_injection_logic}: Dynamic dispatch logic to inject state/rng args.
- {fuzzer_implementation}: The extracted source code of the InputFuzzer class.
- {to_numpy_logic}: Dynamic tensor-to-numpy conversion snippets.
"""

HARNESS_TEMPLATE = r"""
import sys
import importlib.util
import inspect
import traceback
import random
import os
import re
import math
from typing import Any, List, Dict, Tuple, Optional, Union
import numpy as np

# --- DYNAMIC FRAMEWORK IMPORTS ---
{imports}

# --- HELPERS FOR STATE INJECTION ---
{init_helpers}

# --- INJECTED FUZZER LOGIC ---

{fuzzer_implementation}

# -----------------------------

def to_numpy(obj):
    try:
        import numpy as np
    except ImportError:
        return obj

    # --- DYNAMIC TENSOR CONVERSION ---
    {to_numpy_logic}

    # --- GENERIC CONTAINER RECURSION ---
    if isinstance(obj, (list, tuple)):
         return [to_numpy(x) for x in obj]
    return obj

# --- HARNESS LOGIC ---

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        print(f"❌ Could not load module {{name}} at {{path}}")
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"❌ Error during import of {{name}}: {{e}}")
        return None
    return mod

def run_verification(source_path, target_path, source_fw, target_fw, hints_json_str):
    print(f"🧪 Verifying: {{source_path}} vs {{target_path}}")

    mod_src = load_module_from_path("mod_src", source_path)
    mod_tgt = load_module_from_path("mod_tgt", target_path)

    if not mod_src or not mod_tgt:
        print("❌ Module loading failed. Aborting.")
        sys.exit(1)

    all_hints = {{}}
    if hints_json_str:
        try:
            import json
            all_hints = json.loads(hints_json_str)
        except:
            pass

    # Use the injected class alias
    fuzzer = StandaloneFuzzer()
    functions = inspect.getmembers(mod_src, inspect.isfunction)

    passes = 0
    failures = 0
    skips = 0

    import numpy as np

    for func_name, src_func in functions:
        if func_name.startswith("_"):
            continue

        if not hasattr(mod_tgt, func_name):
            print(f"⚠️  Skipping {{func_name}}: Not found in target module.")
            skips += 1
            continue

        tgt_func = getattr(mod_tgt, func_name)

        try:
            sig = inspect.signature(src_func)
            params = list(sig.parameters.keys())
        except ValueError:
            print(f"⚠️  Skipping {{func_name}}: Could not inspect signature.")
            skips += 1
            continue

        func_hints = all_hints.get(func_name, {{}})
        inputs = fuzzer.generate_inputs(params, hints=func_hints)

        # Adapt Inputs
        try:
            src_inputs = fuzzer.adapt_to_framework(inputs, source_fw)
            tgt_inputs = fuzzer.adapt_to_framework(inputs, target_fw)
        except Exception as e:
            print(f"⚠️  Skipping {{func_name}}: Input adaptation failed ({{e}})")
            # traceback.print_exc()
            skips += 1
            continue

        # --- AUTO-INJECTION ---
        try:
            tgt_sig = inspect.signature(tgt_func)
            tgt_params = list(tgt_sig.parameters.keys())

            for tp in tgt_params:
                if tp not in tgt_inputs:
                    # Dynamic Injection Block
                    {param_injection_logic}
                    pass
        except ValueError:
            pass

        # Execute
        try:
            res_src = src_func(**src_inputs)
            res_tgt = tgt_func(**tgt_inputs)

            val_src = to_numpy(res_src)
            val_tgt = to_numpy(res_tgt)

            def deep_compare(a, b):
                # 1. Null check
                if a is None or b is None:
                    return a is b

                # 2. Numpy Logic with Shape & NaN check
                if hasattr(a, "shape") and hasattr(b, "shape"):
                    a_arr = np.asanyarray(a)
                    b_arr = np.asanyarray(b)
                    if a_arr.shape != b_arr.shape:
                        return False
                    # Allow fuzzy match with NaNs treated as equal
                    if a_arr.dtype.kind in ['f', 'c']:
                        return np.allclose(a_arr, b_arr, rtol=1e-3, atol=1e-4, equal_nan=True)
                    return np.array_equal(a_arr, b_arr)

                # 3. Recursion
                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    if len(a) != len(b): return False
                    return all(deep_compare(x, y) for x, y in zip(a, b))

                return a == b

            if deep_compare(val_src, val_tgt):
                print(f"✅ {{func_name}}: Match")
                passes += 1
            else:
                print(f"❌ {{func_name}}: Mismatch")
                # Print debug info on mismatch
                # print(f"Src: {{val_src}}")
                # print(f"Tgt: {{val_tgt}}")
                failures += 1

        except Exception as e:
            print(f"❌ {{func_name}}: Runtime Error ({{e}})")
            # traceback.print_exc()
            failures += 1

    print("-" * 30)
    print(f"📊 Summary: {{passes}} Passed, {{failures}} Failed, {{skips}} Skipped")
    if failures > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    run_verification(
        source_path=r"{source_path}",
        target_path=r"{target_path}",
        source_fw="{source_fw}",
        target_fw="{target_fw}",
        hints_json_str=r'{hints_json}'
    )
"""
