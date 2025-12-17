#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
# ==============================================================================

set -e  # Exit on error

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ROOT_DIR="$(dirname -- "$(dirname -- "$(realpath -- "$0")")")"
VENV_DIR="${VENV_DIR:-$ROOT_DIR}"
TEMP_DIR="_bootstrap_temp"
SEMANTICS_DIR="$ROOT_DIR"'/src/ml_switcheroo/semantics'
SNAPSHOTS_DIR="$ROOT_DIR"'/src/ml_switcheroo/snapshots'

echo "${BOLD}🔥 Initiating Knowledge Base Bootstrap${NC}"
echo "----------------------------------------"

# ------------------------------------------------------------------------------
# 0. DESTRUCTIVE CLEANUP
# ------------------------------------------------------------------------------
echo "${BLUE}[0/5] Cleaning existing JSON artifacts...${NC}"
rm -f "$SEMANTICS_DIR"/k_*.json
rm -f "$SNAPSHOTS_DIR"/*_map.json
echo "   ${GREEN}✔ Cleaned JSONs${NC}"

# 0b. Prepare Workspace
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# ------------------------------------------------------------------------------
# 1. IMPORT TIER A: MATH (Array API Standard)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[1/5] Importing Tier A (Array API)...${NC}"
git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1
echo "   Parsing stubs..."
ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# ------------------------------------------------------------------------------
# 2. IMPORT TIER B: NEURAL (ONNX Operators)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[2/5] Importing Tier B (ONNX Neural)...${NC}"
git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1
echo "   Parsing Operators.md..."
ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# ------------------------------------------------------------------------------
# 3. INTERSECTION DISCOVERY (THE FIX)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[3/5] Auto-Discovering Missing Layers (Linear, ReLU)...${NC}"

# Ensure environment has libraries installed
if [ -d "$VENV_DIR"'/.venv-pyenv-3-12' ]; then
    . "$VENV_DIR"'/.venv-pyenv-3-12/bin/activate'
    python3 -m pip install -e . > /dev/null 2>&1
    # Adding flax explicitly
    python3 -m pip install torch jax flax numpy tensorflow keras mlx > /dev/null 2>&1
else
    # Fallback / Generic
    if command -v uv >/dev/null 2>&1; then
        uv pip install -e . > /dev/null 2>&1
        uv pip install torch jax flax numpy tensorflow keras mlx > /dev/null 2>&1
    else
        python3 -m pip install -e . > /dev/null 2>&1
        python3 -m pip install torch jax flax numpy tensorflow keras mlx > /dev/null 2>&1
    fi
fi

# Run Layer Discovery via integrated CLI (Previously discover_missing_ops.py)
ml_switcheroo discover-layers

# ------------------------------------------------------------------------------
# 4. IMPORT TIER C: INTERNAL
# ------------------------------------------------------------------------------
echo "\n${BLUE}[4/5] Importing Internal Extras...${NC}"
ml_switcheroo import-spec internal

# ------------------------------------------------------------------------------
# 5. IMPLEMENTATION SYNC
# ------------------------------------------------------------------------------
echo "\n${BLUE}[5/5] Syncing Framework Mappings...${NC}"

# Iterate frameworks. IMPORTANT: flax_nnx must be synced to generate the map for the test.
for fw in torch jax flax_nnx numpy tensorflow keras mlx; do
    printf '   👉 Syncing %s... ' "$fw"
    if ml_switcheroo sync "$fw" > /dev/null 2>&1; then
        echo "${GREEN}✔ Done${NC}"
    else
        echo "${RED}Skipped${NC}"
    fi
done

# PaxML Special Case (Python 3.10)
if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ] || [ -d "$VENV_DIR"'/.venv-uv-3-10' ]; then
    printf '   👉 Syncing paxml... '
    if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ]; then
        . "$VENV_DIR"'/.venv-pyenv-3-10/bin/activate'
        python3 -m pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1
    else
        . "$VENV_DIR"'/.venv-uv-3-10/bin/activate'
        uv pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1
    fi
    if ml_switcheroo sync paxml > /dev/null 2>&1; then
        echo "${GREEN}✔ Done${NC}"
    else
        echo "${RED}Failed${NC}"
    fi
else
     echo "${YELLOW}Skipping paxml${NC}"
fi

# ------------------------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------------------------
echo "\n${BOLD}🧹 Cleanup...${NC}"
rm -rf "$TEMP_DIR"

echo "----------------------------------------"
echo "${GREEN}✅ Bootstrap Complete.${NC}"
echo "   Specs: $SEMANTICS_DIR"
echo "   Maps:  $SNAPSHOTS_DIR"
