#!/bin/bash

python3 -c 'import numpy'
python3 -c 'import scipy'
python3 -c 'import pycuda'
python3 -c 'import matplotlib'
python3 -c 'import PIL'
python3 -c 'import cmocean'

for i in $(ls at_*.py); do
    echo "=== Execute $i ==="
    python3 "$i"
    if [ $? != 0 ]; then
        break
    fi
    echo
done
