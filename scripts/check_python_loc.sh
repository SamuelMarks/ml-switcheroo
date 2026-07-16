#!/usr/bin/env bash

# Fail if any python file exceeds 400 lines of code

MAX_LOC=1500
FAILED=0

for file in "$@"; do
    if [[ "$file" == *.py ]]; then
        loc=$(wc -l < "$file" | tr -d ' ')
        if [ "$loc" -gt "$MAX_LOC" ]; then
            echo "Error: $file exceeds $MAX_LOC lines of code ($loc lines). Please refactor to maintain the 'Data over Code' mandate."
            FAILED=1
        fi
    fi
done

exit $FAILED
