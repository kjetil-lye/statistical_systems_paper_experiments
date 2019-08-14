#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:53:48 2019

@author: kjetil
"""

from wasserstein_distance import *


def plotWassersteinConvergencePerturbations(name, basename, r, perturbations):
    plot_info.console_log_show(name)
    wasserstein2pterrors = []
    for (n, p) in enumerate(perturbations[:-1]):
        filename = basename.format(perturbation=p)
        filename_coarse = basename.format(perturbation=perturbations[-1])
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename, k)
            d2 = load(filename_coarse, k)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(perturbations[1:], wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Perturbation $\\epsilon$")
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x, \\epsilon}, \\nu^{2,\\Delta x, \\epsilon_0})||_{L^1(D\\times D)}$')
    plt.title("Wasserstein convergence for %s\nfor second correlation measure,\nwith respect to perturbation size\nagainst a reference solution with $\epsilon_0=%.4f$"%(name,perturbations[-1]))
    showAndSave('%s_wasserstein_perturbation_convergence_2pt_all_components' % name)
    
    
    
    # one point
    wasserstein1pterrors = []
    for (n, p) in enumerate(perturbations[:-1]):
        filename = basename.format(perturbation=p)
        filename_coarse = basename.format(perturbation=perturbations[-1])
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename, k)
            d2 = load(filename_coarse, k)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(perturbations[1:], wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Perturbation $\\epsilon$")
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x, \\epsilon}, \\nu^{1,\\Delta x, \\epsilon_0})||_{L^1(D)}$')
    plt.title("Wasserstein convergence for %s\nfor first correlation measure,\nwith respect to perturbation size\nagainst a reference solution with $\epsilon_0=%.4f$"%(name,perturbations[-1]))
    showAndSave('%s_wasserstein_perturbation_convergence_1pt_all_components' % name)
    
    

def plotWassersteinConvergenceDifferentTypes(name, filenames, r, perturbations_inverse):
    plot_info.console_log_show(f"types {name}")
    wasserstein2pterrors = []
    types = [k for k in filenames.keys()]
    
    if len(types)!=2:
        raise Exception("Only support two perturbation types")
    for filename_a, filename_b in zip(filenames[types[0]], filenames[types[1]]):
        
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename_a, k)
            d2 = load(filename_b, k)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(1.0/np.array(perturbations_inverse,dtype=np.float64), wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Perturbation $\\epsilon$")
    
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x, \\epsilon}_{\\mathrm{%s}}, \\nu^{2,\\Delta x, \\epsilon}_{\\mathrm{%s}})||_{L^1(D\\times D)}$' % (types[0], types[1]))
    plt.title("Wasserstein convergence for %s\nfor second correlation measure"%(name))
    showAndSave('%s_type_comparison_wasserstein_perturbation_convergence_2pt_all_components' % name)
    
    
    
    
    # one point
    wasserstein1pterrors = []
   
   
    for filename_a, filename_b in zip(filenames[types[0]], filenames[types[1]]):
        
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename_a, k)
            d2 = load(filename_b, k)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(1.0/np.array(perturbations_inverse,dtype=np.float64), wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Perturbation $\\epsilon$")
    
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x, \\epsilon}_{\\mathrm{%s}}, \\nu^{1,\\Delta x, \\epsilon}_{\\mathrm{%s}})||_{L^1(D)}$' % (types[0], types[1]))
    plt.title("Wasserstein convergence for %s\nfor first correlation measure,\nwith respect to perturbation size"%(name))
    showAndSave('%s_type_comparison_wasserstein_perturbation_convergence_1pt_all_components' % name)
    
    

if __main__ == '__name__':

    resolution = 1024
    
    import argparse

    parser = argparse.ArgumentParser(description="""
Computes the Wasserstein distance for a range of perturbations
        """)
    
    parser.add_argument('--perturbations', type=float, nargs='+',
                        help='List of perturbations to use')

    parser.add_argument('--basename', type=str, required=True,
                        help='Basename (should have \%d in place of resolution if not using "varying_methods").\n'
                        "This should be the path to the files\n")

    parser.add_argument('--name', type=str, required=True,
                        help='Name of the setup (can include latex in $ ...$)')
    
    parser.add_argument('--varying_types', action='store_true', 
                        help='Compute the convergence with two difference types')
    
    parser.add_argument('--types', type=str, nargs='+',
                        help='The list of types to compare for the varying_types options')

    args = parser.parse_args()

    if not args.varying_types:
        plotWassersteinConvergencePerturbations(args.name, 
                                   args.basename,
                                   resolution,
                                   args.perturbations)

        
    elif args.varying_types:
       
        filenames_per_type = {}
        
        for t in args.types:
            filenames_per_type[t] = []
            for p in args.perturbations:
                filenames_per_type[t].append(args.basename.format(t=t,inv=p))

        plotWassersteinConvergenceDifferentTypes(args.name, filenames_per_type, 
                                                 resolution,
                                                 args.perturbations)