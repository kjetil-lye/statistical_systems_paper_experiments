#!/bin/bash
set -e

function submit {
    bsub -n 1 -N -B -W 120:00 -R 'rusage[mem=64000]' \
	 "$@"
}

function submit_resolutions {
    submit \
	 python ../python/wasserstein_distance.py \
	 --basename $1 \
	 --name $2 \
	 --variable $3 \
	 --norm_order $4 \
	 --normalization $5
}

for variable in "rho" "mx" "my" "E" "all";
do
    for norm_order in 1 2;
    do

	for normalization in 'none' "avg_l${norm_order}" "max_l${norm_order}" "max_point" "avg_point" "max" "avg";
	do
	    

	    submit_resolutions \
		${STATISTICAL_RESOLUTIONS}/kh_conv/n%d/kh_1.nc \
		'Kelvin-Helmholtz' \
		$variable \
		$norm_order \
		$normalization

	    submit_resolutions \
		${STATISTICAL_RESOLUTIONS}/rm_conv/n%d/rm_1.nc \
		'Richtmeyer-Meshkov' \
		$variable \
		$norm_order \
		$normalization


	    submit_resolutions \
		${STATISTICAL_RESOLUTIONS}/brownian_conv/n%d/euler_brownian_1.nc \
		'Brownian_motion' \
		$variable \
		$norm_order \
		$normalization


	    submit_resolutions \
		${STATISTICAL_RESOLUTIONS}/fract01_conv/n%d/euler_brownian_1.nc \
		'Fractional_Brownian_motion_H=0.1' \
		$variable \
		$norm_order \
		$normalization


	    submit \
		python ../python/wasserstein_distance.py \
		--basename ${STATISTICAL_KH_RESOLUTIONS_METHODS}/reconst_{t}/nx_{r}/kh_1.nc \
		--name 'Kelvin-Helmholtz_varying_numerical_scheme' \
		--types 'MC' 'WENO2' \
		--varying_methods \
		--norm_order ${norm_order} \
		--variable ${variable} \
		--normalziation ${normalization}

	    submit \
		python ../python/wasserstein_distance_perturbations.py \
		--perturbations 0.09 0.075 0.06 0.05 0.025 0.01 0.0075  0.005 0.0025 \
		--name 'Kelvin-Helmholtz' \
		--basename ${STATISTICAL_KH_PERTS}/kh_perts/q{perturbation}/kh_1.nc \
		--norm_order ${norm_order} \
		--variable ${variable} \
		--normalziation ${normalization}

	    submit \
		python ../python/wasserstein_distance_perturbations.py \
		--perturbations 8 16 32 64 128 256 512 \
		--name 'Kelvin-Helmholtz_Perturbation_comparison' \
		--basename ${STATISTICAL_KH_PERTS_NORMAL_UNIFORM}/dist_{t}/pertinv_{inv}/kh_1.nc \
		--types 'normal' 'uniform' \
		--varying_types \
		--norm_order ${norm_order} \
		--variable ${variable} \
		--normalziation ${normalization}
	done
    done
done

