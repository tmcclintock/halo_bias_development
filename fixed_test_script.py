import aemulus_data as AD
import numpy as np
from cluster_toolkit import bias
import scipy.optimize as op
import pickle
import emcee, os, sys, itertools
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import quad
import matplotlib.pyplot as plt

sfs = AD.scale_factors()
zs = 1./sfs - 1
x = sfs - 0.5

def model_swap(params, args, xi):
    y = np.log10(200)
    a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
    b1,b2 = 0.183, 1.5
    c1 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4)
    c2 = 2.4
    dropped = args['dropped']
    kept = args['kept']
    pars = np.ones((12))
    pars[kept] = params
    if len(kept) != 12:
        pars[dropped] = args['defaults'][dropped]
    a1,a2,b1,b2,c1,c2 = pars[:6] + xi*pars[6:]
    return a1,a2,b1,b2,c1,c2

def lnprior(params, args):
    if any(np.fabs(params) > 10):
        return -np.inf
    return 0

def lnlike(params, args, return_model=False):
    x = args['x'] # a - 0.5
    biases = args['biases']
    icovs = args['icovs']
    lMbins_all = args['lMbins']
    lMarr = args['lMarr']
    nuarrs = args['nuarrs']
    dndlMs = args['dndlMs']
    n_bins = args['n_bins']
    LL = 0
    if return_model:
        models = []
    for i in range(len(x)):
        dndlM = dndlMs[i]
        lMbins = lMbins_all[i]
        nbin = n_bins[i]
        N = len(biases[i])
        a1,a2,b1,b2,c1,c2 = model_swap(params, args, x[i])
        inds = (dndlM > 0) #Fixes the messed up numerical issue when computing dndlm
        dn = dndlM[:-1] - dndlM[1:]
        for j in range(len(inds)-1): #by hand fixes a numerical scatter UP
            dn = dndlM[j] - dndlM[j+1]
            if dn < 0:
                inds[j+1] = False
        b_n = dndlM * bias._bias_at_nu_FREEPARAMS(nuarrs[i],a1,a2,b1,b2,c1,c2)
        #plt.loglog(lMarr[inds], dndlM[inds])
        b_n_spl = IUS(lMarr[inds], b_n[inds])
        b_model = np.array([quad(b_n_spl, lMbins[j,0], lMbins[j,1])[0]/nbin[j] for j in range(N)])
        if return_model:
            models.append(b_model)
        X = biases[i] - b_model
        LL += np.dot(X, np.dot(icovs[i], X))
    #plt.show()
    #exit()
    if return_model:
        return models
    return -0.5*LL

def lnprob(params, args):
    lp = lnprior(params, args)
    if not np.isfinite(lp): return -1e22
    return lp + lnlike(params, args)

def get_args(i):
    if os.path.isfile("./args/args_box%d.p"%i):
        args = pickle.load(open("./args/args_box%d.p"%i, 'rb'))
        print "Using saved args for box%d"%i
        return args
    else:
        raise Exception("Must have args premade.")
    return

def run_bf(args, doprint=False):
    defaults = args['defaults']
    guess = defaults[args['kept']]
    print "Test lnprob() call = %.2e"%lnprob(guess, args)
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)
    if doprint:
        print result
        print "BF saved at \n\t%s"%bfpath
    np.savetxt(bfpath, result.x)
    if not os.path.isfile(args['default_path']):
        np.savetxt(args['default_path'], result.x)
    return result.fun

def plot_bf(i, args, bfpath, savepath=None):
    #cosmo, h, Omega_m, hmf = get_cosmo(i)
    params = np.loadtxt(bfpath)
    import matplotlib.pyplot as plt
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]
    fig, ax = plt.subplots(2, sharex=True)
    bmodels = lnlike(params, args, return_model=True)
    for j in range(len(zs)):
        bmodel = bmodels[j]
        z = zs[j]
        M = args['Ms'][j]
        b = args['biases'][j]
        be = args['berrs'][j]

        ax[0].errorbar(M, b, be, c=colors[j], marker='.', ls='',label=r"$z=%.2f$"%z)
        ax[0].loglog(M, bmodel, ls='-', c=colors[j])
        pd = (b-bmodel)/bmodel
        pde = be/bmodel
        ax[1].errorbar(M, pd, pde, c=colors[j])
    ax[1].axhline(0, c='k', ls='--', zorder=-1)
    xlim = ax[1].get_xlim()
    ax[1].fill_between(xlim,-.01,.01, color='gray', alpha=0.4, zorder=-2)
    ax[1].set_xlim(xlim)
    ax[0].set_xscale('log')
    ax[0].legend(frameon=False, loc="upper right", fontsize=8)
    ax[1].set_xlabel(r"$M\ [{\rm M_\odot}/h]$")
    ax[1].set_ylabel(r"$(b_{\rm sim}-b_{\rm Fit})/b_{\rm Fit}$")
    ax[0].set_ylabel(r"$b(M)$")
    ax[0].set_yscale('linear')
    plt.subplots_adjust(hspace=0, bottom=0.15, left=0.15)
    if savepath:
        plt.gcf().savefig(savepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    return

def run_mcmc(args, bfpath, mcmcpath, likespath):
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    nwalkers, nsteps = 48, 2000
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=2)
    print "Running MCMC for model:\n\t%s"%(args['name'])
    print "Using fits from:\n\t%s"%bfpath
    sampler.run_mcmc(pos, nsteps)
    print "Saving chain at:\n\t%s"%mcmcpath
    chain = sampler.flatchain
    np.save(mcmcpath, chain)
    likes = sampler.flatlnprobability
    np.save(likespath, likes)
    return
    
if __name__ == "__main__":
    inds = np.arange(12)
    nparams = [4]
    for i in range(len(nparams)):
        npars = nparams[i]
        model_ll_path = "model_evals/bnp%d_loglikes.txt"%npars
        combos = itertools.combinations(inds, 12-npars)
        lls = np.ones(len(list(combos)))*1e99
        startindex = 0
        print "Starting analysis:\n\tNparams = %d\n\tNmodels = %d"%(npars, len(lls))
        if os.path.isfile(model_ll_path):
            lls = np.loadtxt(model_ll_path)
            startindex = np.argmax(lls)
            bestindex = np.argmin(lls)
        model_index = -1
        combos = itertools.combinations(inds, 12-npars)
        for combo in combos:
            model_index += 1
            if npars == 12:
                if model_index > 0:
                    continue
            if model_index != bestindex:
                continue
            else:
                print "Working with the best model at index%d"%bestindex
            lo = 23
            hi = lo+1
            ll = 0 #log likelihood
            for box in range(lo, hi):
                kept = np.delete(inds, combo)
                args = get_args(box)
                args['dropped'] = [ combo[k] for k in range(len(combo))]
                args['kept'] = kept
                args['name'] = "bnp%d_mi%d"%(npars,model_index)
                args['default_path'] = "defaults/defaults_np8_mi0.txt"
                bfpath = "bfs/bf_%s_box%d.txt"%(args['name'], box)
                args['bfpath'] = bfpath
                mcmcpath = "chains/chain_%s_box%d"%(args['name'], box)
                likespath = "chains/likes_%s_box%d"%(args['name'], box)
                if os.path.isfile(args['default_path']):
                    args['defaults'] = np.loadtxt(args['default_path'])
                    print "Using already-made defaults"
                else:
                    y = np.log10(200)
                    a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
                    b1,b2 = 0.183, 1.5
                    c1,c2 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4), 2.4
                    args['defaults'] = np.array([1.6, 1.86, 6.05, 2.34, -5.28, 2.386, 0.0, -1.83, -0.703, 0.0, 0.725, 0.0])
                #ll += run_bf(args, doprint=True)
                #plot_bf(box, args, bfpath)#, "figs/bf_%s_box%d.png"%(args['name'],box))
                run_mcmc(args, bfpath, mcmcpath, likespath)
            print "Np%d Mi%d:\tlnlike = %e"%(npars, model_index, ll)
            lls[model_index] = ll
            #np.savetxt(model_ll_path,lls)
            #exit()
