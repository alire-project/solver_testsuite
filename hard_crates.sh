#!/bin/bash

# Hard crates to solve

set -o errexit

# Check no skipped crates in lace demo solution, and proper hint given
echo lace_gel_full_demo
alr show --tree lace_gel_full_demo | grep -v "missed:skipped" > /dev/null
alr show --tree lace_gel_full_demo | grep "libfreetype^2 (indirect,hinted)" > /dev/null

# Check quick solving of AWA
echo awa
alr show --tree awa > /dev/null

echo OK
