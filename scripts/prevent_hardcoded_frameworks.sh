#!/usr/bin/env bash

# Greps for hardcoded framework routing logic in python files

FAILED=0

for file in "$@"; do
    if [[ "$file" == *.py ]]; then
        if grep -q -E "(if|elif) (framework|target|source|target_fw|source_fw) == (str\()?['\"](jax|torch|mlx|tensorflow|keras|numpy)['\"](\))?:" "$file"; then
            echo "Error: Hardcoded framework routing found in $file"
            grep -n -E "(if|elif) (framework|target|source|target_fw|source_fw) == (str\()?['\"](jax|torch|mlx|tensorflow|keras|numpy)['\"](\))?:" "$file"
            echo "Please use JSON variants or traits for dispatch logic."
            FAILED=1
        fi
    fi
done

exit $FAILED
