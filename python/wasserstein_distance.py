# %load defaults.ipy
import numpy as np
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 600

import matplotlib.pyplot as plt

import sys
sys.path.append('../python')
from plot_info import showAndSave, savePlot, get_environment
import plot_info

import netCDF4
from IPython.core.display import display, HTML
import tikzplotlib
import os
import h5py
import ot
import sys
import scipy
import scipy.stats
import re

conserved_variables = ['rho', 'mx', 'my', 'E']
number_of_variables = len(conserved_variables)

def load(f, sample):
    if '.nc' in f:
        with netCDF4.Dataset(f) as d:
            data = np.zeros((*d.variables[f'sample_{sample}_rho'][:,:,0].shape, number_of_variables))
            for n, variable in enumerate(conserved_variables):
                data[:,:,n] = d.variables[f'sample_{sample}_{variable}'][:,:,0]
    else:
        raise Exception(f"Unsupported file type: {f}")
        
    return data

def normalize(data_at_point, all_data, normalization):
    if normalization.lower() == 'none':
        return data_at_point
    elif re.search(r'avg_l\d+', normalization):
        new_data = np.zeros_like(data_at_point)
        p = int(re.search(r'avg_l(\d+)', normalization).group(1))
        for component in range(all_data.shape[4]):
            normalization_factor = np.mean((np.sum(abs(all_data[:,:,:,component])**p, axis=(0,1)),
                                            )**(1.0/p))
            new_data[:,component] = data_at_point[:, component] / normalization_factor
            
        return new_data
    
    elif re.search(r'max_l\d+', normalization):
        new_data = np.zeros_like(data_at_point)
        p = int(re.search(r'max_l(\d+)', normalization).group(1))
        for component in range(all_data.shape[4]):
            normalization_factor = np.max((np.sum(abs(all_data[:,:,:,component])**p, axis=(0,1)),
                                            )**(1.0/p))
            new_data[:,component] = data_at_point[:, component] / normalization_factor
            
        return new_data
    elif normalization == 'avg_point':
        new_data = np.zeros_like(data_at_point)
        for component in range(all_data.shape[4]):
            normalization = np.mean(data_at_point[:,component])
            new_data[:,component] = data_at_point[:,component]/normalization_factor
        return new_data
    elif normalization == 'max_point':
        new_data = np.zeros_like(data_at_point)
        for component in range(all_data.shape[4]):
            normalization = np.max(abs(data_at_point[:,component]))
            new_data[:,component] = data_at_point[:,component]/normalization_factor
        return new_data
    elif normalization == 'avg':
        new_data = np.zeros_like(data_at_point)
        for component in range(all_data.shape[4]):
            normalization = np.mean(abs(all_data[:,:,:,component]))
            new_data[:,component] = data_at_point[:,component]/normalization_factor
        return new_data
    
    elif normalization == 'max':
        new_data = np.zeros_like(data_at_point)
        for component in range(all_data.shape[4]):
            normalization = np.max(abs(all_data[:,:,:,component]))
            new_data[:,component] = data_at_point[:,component]/normalization_factor
        return new_data
    else:
        raise Exception(f"Unknown normalization {normalization}")

def wasserstein_point2_fast(data1, data2, i, j, ip, jp, a, b, xs, xt, norm_order=2, 
                            normalization='none'):
    """
    Computes the Wasserstein distance for a single point in the spatain domain
    """
    
    

    xs[:,:number_of_variables] = normalize(data1[i,j,:,:], data1, normalization)
    xs[:,number_of_variables:] = normalize(data1[ip, jp, :,:], data1, normalization)

    xt[:,:number_of_variables] = normalize(data2[i,j, :, :], data2, normalization)
    xt[:,number_of_variables:] = normalize(data2[ip, jp, :, :], data2, normalization)


    M = scipy.linalg.norm(xs-xt, ord=norm_order, axis=1)#ot.dist(xs, xt, metric='euclidean')
    G0 = ot.emd(a,b,M)

    return np.sum(G0*M)

def wasserstein1pt_fast(data1, data2, norm_order=2, normalization='none'):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    distance = 0
    
    a = np.ones(N)/N
    b = np.ones(N)/N
    
    for i in range(N):
        for j in range(N):
            xs = normalize(data1[i,j,:,:], data1, normalization)
            xt = normalize(data2[i,j,:,:], data2, normalization)
            M = scipy.linalg.norm(xs-xt, ord=norm_order, axis=1)
            G0 = ot.emd(a,b,M)

            distance += np.sum(G0*M)


    
    return distance / N**2



def wasserstein2pt_fast(data1, data2, norm_order=2, normalization='none'):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N, 2*number_of_variables))
    xt = np.zeros((N, 2*number_of_variables))
    distance = 0
    
    number_of_integration_points = 16
    points = np.arange(number_of_integration_points)
    for (n,x) in enumerate(points):
        for y in points:

            for xp in points:
                for yp in points:
                    i = int(x*N/number_of_integration_points)
                    j = int(y*N/number_of_integration_points)
                    ip = int(xp*N/number_of_integration_points)
                    jp = int(yp*N/number_of_integration_points)
                    distance += wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt, norm_order=norm_order, normalization=normalization)


    
    return distance / len(points)**4



def plotWassersteinConvergence(name, basename, resolutions, norm_order=2, normalization='none'):
    plot_info.console_log_show(f"types {name}")
    wasserstein2pterrors = []
    for r in resolutions[1:]:
        filename = basename % r
        filename_coarse = basename % int(r/2)
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename, k)
            d2 = np.repeat(np.repeat(load(filename_coarse, k), 2,0), 2,1)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2, norm_order=norm_order, normalization=normalization))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(resolutions[1:], wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions[1:], ['${r} \\times {r}$'.format(r=r) for r in resolutions[1:]])
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x}, \\nu^{2,\\Delta x/2})||_{L^1(D\\times D)}$')
    plt.title(f"Wasserstein convergence for {name}\nfor second correlation marginal\n{conserved_variables}\n$\\ell^{{{norm_order}}}$, normalization: {normalization}")
    if number_of_variables > 1:
        showAndSave('%s_l_%d_%s_wasserstein_convergence_2pt_all_components' % (name, norm_order, normalization))
    else:
        showAndSave('%s_l_%d_%s_wasserstein_convergence_2pt_%s' % (name, norm_order, normalization, conserved_variables[0]))
        
    
    
    wasserstein1pterrors = []
    for r in resolutions[1:]:
        filename = basename % r
        filename_coarse = basename % int(r/2)
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename, k)
            d2 = np.repeat(np.repeat(load(filename_coarse, k), 2,0), 2,1)
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2, norm_order=norm_order, normalization=normalization))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(resolutions[1:], wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions[1:], ['${r} \\times {r}$'.format(r=r) for r in resolutions[1:]])
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x}, \\nu^{1, \\Delta x/2})||_{L^1(D)}$')
    plt.title(f"Wasserstein convergence for {name}\nfor first correlation marginal\n{conserved_variables}\n$\\ell^{{{norm_order}}}$, normalization: {normalization}")

    if number_of_variables > 1:
        showAndSave('%s_l_%d_%s_wasserstein_convergence_1pt_all_components' % (name, norm_order, normalization))
    else:
        showAndSave('%s_l_%d_%s_wasserstein_convergence_1pt_%s' % (name, norm_order, normalization, conserved_variables[0]))
        
    


def plotWassersteinConvergenceVaryingMethods(name, filenames, resolutions, norm_order=2, normalization='none'):
    plot_info.console_log_show(f"methods {name}")
    # two point
    wasserstein2pterrors = []
    types = [t for t in filenames.keys()]
    for n, (filename_a, filename_b) in enumerate(zip(filenames[types[0]], filenames[types[1]])):
        r = resolutions[n]
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename_a, k)
            d2 = load(filename_b, k)
           
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2, norm_order=norm_order, normalization=normalization))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(resolutions, wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions, ['${r} \\times {r}$'.format(r=r) for r in resolutions])
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x}_{\\mathrm{%s}}, \\nu^{2,\\Delta x}_{\\mathrm{%s}})||_{L^1(D\\times D)}$' %(types[0], types[1]))
    plt.title(f"Wasserstein convergence for {name}\nfor second correlation marginal\n{conserved_variables}\n$\\ell^{{{norm_order}}}$, normalization: {normalization}")

    if number_of_variables > 1:
        showAndSave('%s_l_%d_%s_scheme_wasserstein_convergence_2pt_all_components' % (name, norm_order, normalization))
    else:
        showAndSave('%s_l_%d_%s_scheme_wasserstein_convergence_2pt_%s' % (name, norm_order, normalization, conserved_variables[0]))
    # one point
    wasserstein1pterrors = []
    types = [t for t in filenames.keys()]
    for n, (filename_a, filename_b) in enumerate(zip(filenames[types[0]], filenames[types[1]])):
        r = resolutions[n]
        
        data1 = np.zeros((r,r,r, number_of_variables))
        data2 = np.zeros((r,r,r, number_of_variables))
        for k in range(r):
            d1 = load(filename_a, k)
            d2 = load(filename_b, k)
            
           
            data1[:,:,k,:] = d1
            data2[:,:,k,:] = d2

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2, norm_order=norm_order, normalization=normalization))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(resolutions, wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions, ['${r} \\times {r}$'.format(r=r) for r in resolutions])
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x}_{\\mathrm{%s}}, \\nu^{1,\\Delta x/2}_{\\mathrm{%s}})||_{L^1(D)}$' %(types[0], types[1]))
    plt.title(f"Wasserstein convergence for {name}\nfor first correlation marginal\n{conserved_variables}\n$\\ell^{{{norm_order}}}$, normalization: {normalization}")
    if number_of_variables > 1:
        showAndSave('%s_l_%d_%s_scheme_wasserstein_convergence_1pt_all_components' % (name, norm_order, normalization))
    else:
        showAndSave('%s_l_%d_%s_scheme_wasserstein_convergence_1pt_%s' % (name, norm_order, normalization, conserved_variables[0]))

if __name__ == '__main__':
    resolutions = [64, 128, 256, 512, 1024]
    
    import argparse

    parser = argparse.ArgumentParser(description="""
Computes the Wasserstein distance for a range of resolutions
        """)

    parser.add_argument('--basename', type=str, required=True,
                        help='Basename (should have \%d in place of resolution if not using "varying_methods").\n'
                        "This should be the path to the files\n")

    parser.add_argument('--name', type=str, required=True,
                        help='Name of the setup (can include latex in $ ...$)')
    
    parser.add_argument('--varying_methods', action='store_true', 
                        help='Compute the convergence with two difference methods')
    
    parser.add_argument('--types', type=str, nargs='+',
                        help='The list of types to compare for the varying_method options')

    parser.add_argument('--variable', type=str, default='all',
                        help='The variable to compute convergence (rho, mx, my, E or all)')

    parser.add_argument('--norm_order', type=int, default=2,
                        help='The order of the l^p norm for the distance on R^(kN)')

    parser.add_argument('--normalization', type=str, default='none',
                        help="The normalization to apply to the samples. Available options: none, avg_l<p>, max_l<p>, avg_point, max_point, avg, max")
    

    args = parser.parse_args()

    if args.variable.lower() != 'all':
        conserved_variables = [args.variable]
        number_of_variables = 1

    if not args.varying_methods:
        plotWassersteinConvergence(args.name, args.basename, resolutions, 
                                   norm_order=args.norm_order, 
                                   normalization=args.normalization)
        
    elif args.varying_methods:
        
        filenames_per_type = {}

        for t in args.types:
            filenames_per_type[t] = []
            for r in resolutions:
                filenames_per_type[t].append(args.basename.format(t=t,r=r))

        plotWassersteinConvergenceVaryingMethods(args.name, filenames_per_type, 
                                                 resolutions, norm_order=norm_order,
                                                 normalization=args.normalization)
