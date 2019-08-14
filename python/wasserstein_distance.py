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

def wasserstein_point2_fast(data1, data2, i, j, ip, jp, a, b, xs, xt):
    """
    Computes the Wasserstein distance for a single point in the spatain domain
    """
    

    xs[:,:number_of_variables] = data1[i,j,:,:]
    xs[:,number_of_variables:] = data1[ip, jp, :,:]

    xt[:,:number_of_variables] = data2[i,j, :, :]
    xt[:,number_of_variables:] = data2[ip, jp, :, :]


    M = ot.dist(xs, xt, metric='euclidean')
    G0 = ot.emd(a,b,M)

    return np.sum(G0*M)

def wasserstein1pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    distance = 0
    
    a = np.ones(N)/N
    b = np.ones(N)/N
    
    for i in range(N):
        for j in range(N):
            xs = data1[i,j,:,:]
            xt = data2[i,j,:,:]
            M = ot.dist(xs, xt, metric='euclidean')
            G0 = ot.emd(a,b,M)

            distance += np.sum(G0*M)


    
    return distance / N**2



def wasserstein2pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N, 2*number_of_variables))
    xt = np.zeros((N, 2*number_of_variables))
    distance = 0
    

    points = 0.1*np.array(range(0,10))
    for (n,x) in enumerate(points):
        for y in points:

            for xp in points:
                for yp in points:
                    i = int(x*N)
                    j = int(y*N)
                    ip = int(xp*N)
                    jp = int(yp*N)
                    distance += wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt)


    
    return distance / len(points)**4



def plotWassersteinConvergence(name, basename, resolutions):
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

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(resolutions[1:], wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions[1:], ['${r} \\times {r}$'.format(r=r) for r in resolutions[1:]])
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x}, \\nu^{2,\\Delta x/2})||_{L^1(D\\times D)}$')
    plt.title("Wasserstein convergence for %s\nfor second correlation marginal"%name)
    showAndSave('%s_wasserstein_convergence_2pt_all_components' % name)
    
    
    
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

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(resolutions[1:], wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions[1:], ['${r} \\times {r}$'.format(r=r) for r in resolutions[1:]])
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x}, \\nu^{1, \\Delta x/2})||_{L^1(D)}$')
    plt.title("Wasserstein convergence for %s\nfor first correlation marginal"%name)
    showAndSave('%s_wasserstein_convergence_1pt_all_components' % name)
    
    


def plotWassersteinConvergenceVaryingMethods(name, filenames, resolutions):
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

        wasserstein2pterrors.append(wasserstein2pt_fast(data1, data2))
        print("wasserstein2pterrors=%s" % wasserstein2pterrors)
    

    plt.loglog(resolutions, wasserstein2pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions, ['${r} \\times {r}$'.format(r=r) for r in resolutions])
    plt.ylabel('$||W_1(\\nu^{2, \\Delta x}_{\\mathrm{%s}}, \\nu^{2,\\Delta x}_{\\mathrm{%s}})||_{L^1(D\\times D)}$' %(types[0], types[1]))
    plt.title("Wasserstein convergence for %s\nfor second correlation marginal"%name)
    showAndSave('%s_scheme_wasserstein_convergence_2pt_all_components' % name)
    
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

        wasserstein1pterrors.append(wasserstein1pt_fast(data1, data2))
        print("wasserstein1pterrors=%s" % wasserstein1pterrors)
    

    plt.loglog(resolutions, wasserstein1pterrors, '-o', basex=2, basey=2)
    plt.xlabel("Resolution")
    plt.xticks(resolutions, ['${r} \\times {r}$'.format(r=r) for r in resolutions])
    plt.ylabel('$||W_1(\\nu^{1, \\Delta x}_{\\mathrm{%s}}, \\nu^{1,\\Delta x/2}_{\\mathrm{%s}})||_{L^1(D)}$' %(types[0], types[1]))
    plt.title("Wasserstein convergence for %s\nfor first correlation marginal"%name)
    showAndSave('%s_scheme_wasserstein_convergence_1pt_all_components' % name)

if __main__ == '__name__':
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

    args = parser.parse_args()

    if not args.varying_methods:
        plotWassersteinConvergence(args.name, args.basename, resolutions)
        
    elif args.varying_methods:
        
        filenames_per_type = {}

        for t in args.types:
            filenames_per_type[t] = []
            for r in resolutions:
                filenames_per_type[t].append(args.basename.format(t=t,r=r))

        plotWassersteinConvergenceVaryingMethods(args.name, filenames_per_type, resolutions)