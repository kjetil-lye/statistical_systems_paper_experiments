#!/bin/bash
set -e
$HOME/.local/bin/jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute WassersteinDistancesPerturbationsAll.ipynb --output WassersteinDistancesPerturbationsAllOutput.ipynb
$HOME/.local/bin/jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute WassersteinDistances.ipynb --output WassersteinDistancesOutput.ipynb

