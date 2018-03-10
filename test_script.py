import aemulus_data as AD
from classy import Class
import cluster_toolkit as ct
from cluster_toolkit import bias
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import emcee
sfs = AD.scale_factors()
zs = 1./sfs - 1
x = sfs - 0.5

model_number = 14
name = 'model%d'%model_number

def start(name, xi=None):
    y = np.log10(200)
    a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
    b1,b2 = 0.183, 1.5
    c1,c2 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4), 2.4
    if name == 'all':
        return np.array([a1,a2,b1,b2,c1,c2])
    if name == 'model1':
        return np.array([a1,a2,b1,b2,c1])
    if name == 'model2':
        return np.array([a1,a2,b1,b2,c2])
    if name == 'model3':
        return np.array([a1,a2,b2,c1])
    if name == 'model4':
        return np.array([a2,b2,c1])
    if name == 'model5': #BAD MODEL
        return np.array([a2,b2,c2])
    if name == 'model6':
        return np.array([0.9,-3.37]) #Good start for a2, c1
    if name == 'model7':
        return np.array([0.9,-3.37, 2.4]) #Good start for a2, c1, c2
    if name == 'model8':
        return np.array([0.9,0.0,-3.37, 2.4]) #Good start for a2_0, a2_1, c1, c2
    if name == 'model9':
        return np.array([0.9,0.0, 4.0, 0.0,-3.37, 2.4]) #a2_0, a2_1, b1_0, b1_1, c1, c2
    if name == 'model10':
        return np.array([0.9,0.0, 4.0, 0.0, 2.4]) #a2_0, a2_1, b1_0, b1_1, c2 #FAILS
    if name == 'model11':
        return np.array([0.9,0.0, 4.0, 0.0, 2.4, 0.0]) #a2_0, a2_1, b1_0, b1_1, c2_0, c2_1 #FAILS
    if name == 'model12':
        return np.array([0.9,0.0, 4.0, 0.0,-3.37, 0]) #a2_0, a2_1, b1_0, b1_1, c1_0, c1_1
    if name == 'model13':
        return np.array([0.9,0.0, 4.0, 0.0, -1.0]) #a2_0, a2_1, b1_0, b1_1, c1_1
    if name == 'model14':
        return np.array([0.9,0.0, 4.0, -1.0]) #a2_0, a2_1, b1_0, c1_1

def model_swap(params, name, args, xi=None):
    y = np.log10(200)
    a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
    b1,b2 = 0.183, 1.5
    c1 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4)
    c2 = 2.4
    if name == 'all':
        a1,a2,b1,b2,c1,c2 = params
    if name == 'model1':
        a1,a2,b1,b2,c1 = params
    if name == 'model2':
        a1,a2,b1,b2,c2 = params
    if name == 'model3':
        b1 = 4.0
        a1,a2,b2,c1 = params
    if name == 'model4':
        a1,b1 = 1.6, 4.0
        a2,b2,c1 = params
    if name == 'model5': #BAD MODEL
        a1,b1 = 1.6, 4.0
        a2,b2,c2 = params
    if name == 'model6':
        a1,b1,b2 = 1.6, 4.0, 2.33852598
        a2,c1 = params
    if name == 'model7':
        a1,b1,b2 = 1.6, 4.0, 2.33852598
        a2,c1,c2 = params
    if name == 'model8':
        a1,b1,b2 = 1.6, 4.0, 2.33852598
        a2_0,a2_1,c1,c2 = params
        a2 = a2_0 + args['x'][xi]*a2_1
    if name == 'model9':
        a1, b2 = 1.6, 2.33852598
        a2_0,a2_1,b1_0,b1_1,c1,c2 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
    if name == 'model10':
        a1, b2, c1 = 1.6, 2.33852598, -0.505
        a2_0,a2_1,b1_0,b1_1,c2 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
    if name == 'model11':
        a1, b2, c1 = 1.6, 2.33852598, -0.505
        a2_0,a2_1,b1_0,b1_1,c2_0,c2_1 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
        c2 = c2_0 + args['x'][xi]*c2_1
    if name == 'model12':
        a1, b2, c2 = 1.6, 2.33852598, 2.38569171
        a2_0,a2_1,b1_0,b1_1,c1_0,c1_1 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
        c1 = c1_0 + args['x'][xi]*c1_1
    if name == 'model13':
        a1, b2, c1_0, c2 = 1.6, 2.33852598, -4.2, 2.38569171
        a2_0,a2_1,b1_0,b1_1,c1_1 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
        c1 = c1_0 + args['x'][xi]*c1_1    
    if name == 'model14':
        a1, b1_1, b2, c1_0, c2 = 1.6, 1.2, 2.33852598, -4.2, 2.38569171
        a2_0,a2_1,b1_0,c1_1 = params
        a2 = a2_0 + args['x'][xi]*a2_1
        b1 = b1_0 + args['x'][xi]*b1_1
        c1 = c1_0 + args['x'][xi]*c1_1    
    return a1,a2,b1,b2,c1,c2


def lnprior(params, args):
    a1,a2,b1,b2,c1,c2 = model_swap(params, args['name'], args, 0)
    #if np.fabs(a1) > 10 or np.fabs(c2) > 10:
    #    return -np.inf
    return 0

def lnlike(params, args):
    nus = args['nus']
    biases = args['biases']
    icovs = args['icovs']
    LL = 0
    for i in range(len(nus)):
        a1,a2,b1,b2,c1,c2 = model_swap(params, args['name'], args, i)
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
    return {'nus':nus, 'biases':bs, 'icovs':icovs, 'berrs':bes, 'Ms':Ms, 'name':name, 'x':x}

def run_bf(args, bfpath):
    guess = start(args['name'])
    print "Test lnprob() call = %.2e"%lnprob(guess, args)
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll, guess, args=args)
    print result
    np.savetxt(bfpath, result.x)
    print "BF saved at \n\t%s"%bfpath
    return result.fun

def plot_bf(i, args, bfpath, show=False):
    cosmo, h, Omega_m = get_cosmo(i)
    params = np.loadtxt(bfpath)
    print i, params
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]
    fig, ax = plt.subplots(2, sharex=True)
    for j in range(len(zs)):
        z = zs[j]
        M = args['Ms'][j]
        b = args['biases'][j]
        be = args['berrs'][j]
        nu = args['nus'][j]
        a1,a2,b1,b2,c1,c2 = model_swap(params, args['name'], args, j)
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
    plt.subplots_adjust(hspace=0, bottom=0.15, left=0.15)
    fig.savefig("figs/bias_fit_box%d_model%d.png"%(i,model_number))
    if show:
        plt.show()
    plt.clf()

def run_mcmc(args, bfpath, mcmcpath, likespath):
    bf = np.loadtxt(bfpath)
    ndim = len(bf)
    nwalkers = 48
    nsteps = 4000
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
    lo = 0
    hi = 40
    ll = 0
    for i in range(lo, hi):
        args = get_args(i)
        bfpath = "bfs/bf_%s_box%d_bias.txt"%(args['name'], i)
        mcmcpath = "mcmcs/mcmc_%s_box%d_bias.txt"%(args['name'], i)
        likespath = "mcmcs/likes_%s_box%d_bias.txt"%(args['name'], i)
        ll += run_bf(args, bfpath)
        if np.isnan(ll):
            print "Failure on box %d"%i
            exit()
        #plot_bf(i, args, bfpath, show=True*0)
        run_mcmc(args, bfpath, mcmcpath, likespath)
    print "%s LL total = %e"%(args['name'], ll)
