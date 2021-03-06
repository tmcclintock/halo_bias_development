"""This contains routines to find the best fit and run the MCMC."""

import os, pickle
import numpy as np
import scipy.optimize as op
from likelihoods import *
import swaps

def get_args(i, bname):
    if os.path.isfile("./args/args_%s%d.p"%(bname, i)):
        args = pickle.load(open("./args/args_%s%d.p"%(bname, i), 'rb'))
        print "Using saved args for %s%d"%(bname, i)
        return args
    else:
        raise Exception("Must have args premade.")
    return #won't ever reach this
    
def run_bf(args, bf_array, bf_path):
    box = args['box'] #which simulation we are looking at
    guess = swaps.initial_guess(args)
    print "Test lnprob() call = %.2e"%lnprob(guess, args)
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)
    print result
    bf_array[box] = result['x']
    np.save(bf_path, bf_array)
    return bf_array

def run_mcmc(args, bf_array, mcmc_path, likes_path):
    box = args['box']
    bf = bf_array[box]
    ndim = len(bf)
    nwalkers, nsteps = 2*ndim+4, 2000
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=2)
    print "Running MCMC for model:\n\t%s"%(args['name'])
    print "\tUsing fits for box %d"%box
    sampler.run_mcmc(pos, nsteps)
    print "Saving chain at:\n\t%s"%mcmc_path
    np.save(mcmc_path, sampler.flatchain)
    print "Saving likes at:\n\t%s"%likes_path
    np.save(likes_path, sampler.flatlnprobability)
    return
