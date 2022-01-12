#!/usr/bin/env bash

echo "generating conda env file"
conda env export > environment.yml
echo "generating pip reqauirements file"
poetry export --without-hashes -o requirements.txt
