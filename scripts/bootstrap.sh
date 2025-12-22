#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
#
# This script hydrates the Semantic Knowledge Base (Hub) and Framework
# Implementation Maps (Spokes) from upstream sources and the local environment.
#
# Stages:
# 1. Ingestion: Import external specs (Array API, ONNX).
# 2. Discovery: Find consensus standards across frameworks.
# 3. Scaffolding: Heuristic mapping.
# 4. Snapshots: Capture API signatures for Ghost Mode.
# 5. Sync: Link implementations to abstract standards.
# ==============================================================================

set -e

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
echo "${BLUE}[0/6] Cleaning existing JSON artifacts...${NC}"
rm -f "$SEMANTICS_DIR"/k_*.json
# Cleaning the specific Discovered/Pending standards filename
rm -f "$SEMANTICS_DIR"/k_discovered.json
# Clean Overlays (*_map.json)
rm -f "$SNAPSHOTS_DIR"/*_map.json
# Clean Snapshots (*_v*.json) to ensure fresh capture
rm -f "$SNAPSHOTS_DIR"/*_v*.json
echo "   ${GREEN}✔ Cleaned JSONs${NC}"

# 0b. Prepare Workspace
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# ------------------------------------------------------------------------------
# 1. IMPORT TIER A: MATH (Array API Standard)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[1/6] Importing Tier A (Array API)...${NC}"
git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1
echo "   Parsing stubs..."
ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# ------------------------------------------------------------------------------
# 2. IMPORT TIER B: NEURAL (ONNX Operators)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[2/6] Importing Tier B (ONNX Neural)...${NC}"
git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1
echo "   Parsing Operators.md..."
ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# ------------------------------------------------------------------------------
# 3. INTERSECTION DISCOVERY (THE FIX)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[3/6] Auto-Discovering Standards via Consensus...${NC}"

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

# Run Consensus Discovery via CLI (Generates k_discovered.json)
# We explicitly target layers, activations, losses, and optimizers
ml_switcheroo sync-standards --categories layer activation loss optimizer

# ------------------------------------------------------------------------------
# 3b. HEURISTIC SCAFFOLDING (The Wire-in)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[3b/6] Scaffolding via Heuristics (Regex Mapping)...${NC}"
# Activates the dormant 'discovery_heuristics' logic in FrameworkAdapters.
# Populates snapshots with mappings derived from naming conventions.
# Note: We skip 'flax_nnx' here as it is an adapter key, not a package name valid for inspection.
if [ "$(uname)" = 'Linux' ]; then
  ml_switcheroo scaffold --frameworks torch jax numpy keras mlx
else
  # TODO: Fix tensorflow on Linux
  ml_switcheroo scaffold --frameworks torch jax numpy keras mlx tensorflow
fi
# ------------------------------------------------------------------------------
# 4. IMPORT TIER C: INTERNAL
# ------------------------------------------------------------------------------
echo "\n${BLUE}[4/6] Importing Internal Extras...${NC}"
ml_switcheroo import-spec internal

# ------------------------------------------------------------------------------
# 5. GHOST SNAPSHOT CAPTURE (New)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[5/6] Capturing API Snapshots (Ghost Mode)...${NC}"
# Dumps raw API signatures (e.g. torch_v2.1.0.json) for client-side usage
ml_switcheroo snapshot --out-dir "$SNAPSHOTS_DIR"

# ------------------------------------------------------------------------------
# 6. IMPLEMENTATION SYNC
# ------------------------------------------------------------------------------
echo "\n${BLUE}[6/6] Syncing Framework Mappings...${NC}"

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
