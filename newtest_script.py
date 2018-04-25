import aemulus_data as AD
from classy import Class
import cluster_toolkit as ct
from cluster_toolkit import bias
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import emcee, os, sys, itertools
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
        defaults = args['defaults']
        pars[dropped] = defaults[dropped]
    a1,a2,b1,b2,c1,c2 = pars[:6] + xi*pars[6:]
    return a1,a2,b1,b2,c1,c2

def lnprior(params, args):
    x = args['x']
    for xi in x:
        pars = np.array(model_swap(params, args, xi))
        #if any(pars < 0) or any(pars > 5):
        #    return -np.inf
    return 0

def lnlike(params, args):
    x = args['x'] # a - 0.5
    nus = args['nus']
    biases = args['biases']
    icovs = args['icovs']
    LL = 0
    for i in range(len(nus)):
        a1,a2,b1,b2,c1,c2 = model_swap(params, args, x[i])
        b_model = bias._bias_at_nu_FREEPARAMS(nus[i],a1,a2,b1,b2,c1,c2)
        X = biases[i] - b_model
        LL += np.dot(X, np.dot(icovs[i], X))
    return -0.5*LL

def lnprob(params, args):
    lp = lnprior(params, args)
    if not np.isfinite(lp): return -1e22
    return lp + lnlike(params, args)

def get_cosmo(i):
    obh2, och2, w, ns, ln10As, H0, Neff, s8 = AD.building_box_cosmologies()[i]
    h = H0/100.
    Omega_b = obh2/h**2
    Omega_c = och2/h**2
    Omega_m = Omega_b+Omega_c
    params = {'output': 'mPk', 'h': h, 'ln10^{10}A_s': ln10As, 'n_s': ns, 'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b, 'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m, 'N_eff': Neff, 'P_k_max_1/Mpc':10., 'z_max_pk':10. }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo, h, Omega_m

def get_args(i):
    Ms = []
    bs = []
    bes = []
    icovs = []
    cosmo, h, Omega_m = get_cosmo(i)
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    kh = k/h
    nus = [] #sigma^2
    for j in range(0,10): #snap
        z = zs[j]
        M, b, be = np.loadtxt("linear/Box%03d_Z%d_DS50_linearbias.txt"%(i,j)).T
        M = np.ascontiguousarray(M)
        Ms.append(M)
        bs.append(b)
        bes.append(be)
        p = np.array([cosmo.pk_lin(ki, z) for ki in k])*h**3
        nus.append(ct.bias.nu_at_M(M, kh, p, Omega_m))
        cov = np.diag(be**2)
        icovs.append(np.linalg.inv(cov))
    return {'nus':nus, 'biases':bs, 'icovs':icovs, 'berrs':bes, 'Ms':Ms, 'x':x}

def run_bf(args, doprint=False):
    default_path = args['default_path']
    bfpath = args['bfpath']
    if os.path.isfile(default_path):
        defaults = np.loadtxt(default_path)
    else:
        y = np.log10(200)
        a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
        b1,b2 = 0.183, 1.5
        c1,c2 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4), 2.4
        #defaults = np.array([a1,a2,b1,b2,c1,c2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2])
        defaults = np.array([1.6, 1.86, 6.05, 2.34, -5.28, 2.386, 0.0, -1.83, -0.703, 0.0, 0.725, 0.0]) #model12
        #defaults = np.array([1.6, 2.4584, 4.934, 2.339, -4.2, 2.385, 0.0, -3.014, 1.2, 0.0, -1.1172, 0.0]) #taken from model14 in test_script.py
    args['defaults'] = np.copy(defaults)
    guess = defaults[args['kept']]
    if doprint:
        print "Test lnprob() call = %.2e"%lnprob(guess, args)
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)
    if doprint:
        print result
        print "BF saved at \n\t%s"%bfpath
    np.savetxt(bfpath, result.x)
    if not os.path.isfile(default_path):
        np.savetxt(default_path, result.x)
    return result.fun

def plot_bf(i, args, bfpath, savepath=None):
    cosmo, h, Omega_m = get_cosmo(i)
    params = np.loadtxt(bfpath)
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]
    fig, ax = plt.subplots(2, sharex=True)
    for j in range(len(zs)):
        z = zs[j]
        M = args['Ms'][j]
        b = args['biases'][j]
        be = args['berrs'][j]
        nu = args['nus'][j]
        a1,a2,b1,b2,c1,c2 = model_swap(params, args, args['x'][j])
        bmodel = bias._bias_at_nu_FREEPARAMS(nu,a1,a2,b1,b2,c1,c2)
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
    #ax[0].text(1e13, 4, "PRELIMINARY", color='r', fontsize=20, alpha=0.5)
    #ax[1].text(1e13, 0.02, "PRELIMINARY", color='r', fontsize=20, alpha=0.5)
    plt.subplots_adjust(hspace=0, bottom=0.15, left=0.15)
    if savepath:
        plt.gcf().savefig(savepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

def run_mcmc(args, bfpath, mcmcpath, likespath):
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    nwalkers = 48
    nsteps = 2000
    pos = [bf + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(args,), threads=4)
    print "Running MCMC for model:\n\t%s"%(args['name'])
    print "Using fits from:\n\t%s"%bfpath
    sampler.run_mcmc(pos, nsteps)
    print "Saving chain at:\n\t%s"%mcmcpath
    chain = sampler.flatchain
    np.savetxt(mcmcpath, chain)
    likes = sampler.flatlnprobability
    np.savetxt(likespath, likes)

    
if __name__ == "__main__":
    inds = np.arange(12)
    nparams = [6]
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
            #if model_index < startindex:
            #    continue
            if model_index != bestindex:
                continue
            else:
                print "Working with the best model"
            #print model_index, lls[model_index-1], lls[model_index]
            lo = 0
            hi = 40#lo+1
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
                mcmcpath = "chains/chain_%s_box%d.txt"%(args['name'], box)
                likespath = "chains/likes_%s_box%d.txt"%(args['name'], box)
                ll += run_bf(args, doprint=True*0)
                #plot_bf(box, args, bfpath, "figs/bf_%s_box%d.png"%(args['name'],box))
                run_mcmc(args, bfpath, mcmcpath, likespath)
            print "Np%d Mi%d:\tlnlike = %e"%(npars, model_index, ll)
            lls[model_index] = ll
            #np.savetxt(model_ll_path,lls)
            #exit()
