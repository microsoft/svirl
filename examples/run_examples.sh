#!/bin/bash

for i in $(ls example_*.py); do
    echo "=== Execute $i ==="
    python "$i"
    if [ $? != 0 ]; then
        break
    fi
    echo
done
