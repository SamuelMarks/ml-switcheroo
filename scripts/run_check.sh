#!/bin/sh

# Fail on any error
set -e

echo "🧪 Running Test Suite..."
pytest tests

echo ""
echo "📝 Generating Demo Input..."
mkdir -p examples

# Create a demo file showing different tier capabilities
# Note: We use 'torch.add' with alpha, which is a standard op mapped
# to the 'decompose_alpha' plugin in the semantics.
cat <<EOF > examples/demo.py
import torch

def model_forward(x, y):
    # Tier A (Array API): Standard math swap
    val = torch.abs(x)

    # Tier B (Neural): Complex rewrite via Plugin (alpha decomposition)
    # torch.add with alpha isn't in JAX, so we rewrite to math
    scaled = torch.add(val, y, alpha=0.5)

    # Tier C (Extras): Unknown function -> Escape Hatch
    res = torch.unknown_magic(scaled)

    return res
EOF

echo "🔄 Running ml-switcheroo Transpiler (Strict Mode)..."
# We use --strict to force the Escape Hatch on unknown functions
ml_switcheroo convert 'examples/demo.py' --out 'examples/demo_jax.py' --strict

echo ""
echo "🔍 Verifying Output Against Baseline..."

BASELINE="tests/snapshots/demo_expect.py"
GENERATED="examples/demo_jax.py"

if [ ! -f "$BASELINE" ]; then
    echo "⚠️  Baseline not found at $BASELINE. Creating it now..."
    mkdir -p tests/snapshots
    cp "$GENERATED" "$BASELINE"
    echo "✅ Baseline created. Commit this file!"
else
    # ignore whitespace differences with -w if strict formatting isn't guaranteed
    if diff -w "$BASELINE" "$GENERATED"; then
        echo "✅ Output matches baseline exactly."
    else
        echo "❌ Output differs from baseline!"
        echo "   If this change is intentional, update $BASELINE"
        exit 1
    fi
fi

echo ""
echo "✨ All Checks Passed."
