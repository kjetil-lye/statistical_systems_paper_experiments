#!/bin/bash
set -e
notebook_basename=${1//.ipynb/}
jupyter  nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute ${notebook_basename}.ipynb --output ${notebook_basename}Output.ipynb
