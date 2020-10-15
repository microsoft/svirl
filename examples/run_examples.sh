#!/bin/bash

for i in $(ls example_*.py); do
    echo "=== Execute $i ==="
    python3 "$i"
    if [ $? != 0 ]; then
        break
    fi
    echo
done
