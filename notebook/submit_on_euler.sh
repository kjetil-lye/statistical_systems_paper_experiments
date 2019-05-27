#!/bin/bash
set -e
export OMP_NUM_THREADS=1
for notebook in BV_experiments FractionalBrownianMotion KelvinHelmholtzSingleSample RichtmeyerMeshkovSingleSample StructureAlsvinn WassersteinDistances WassersteinDistancesPerturbationsAll;
do
    if [ ! -f ${notebook}.ipynb ];
    then
	echo "Does not exist: ${notebook}.ipynb"
	exit 1
    fi

    echo bsub -W 24:00 -R 'rusage[mem=8000]' jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute ${notebook}.ipynb --output ${notebook}Output.ipynb
done
