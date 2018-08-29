import aemulus_data as AD
from classy import Class
import cluster_toolkit as ct
from cluster_toolkit import bias
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import pickle
#import aemHMF
import emcee, os, sys, itertools
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import quad

sfs = AD.scale_factors()
zs = 1./sfs - 1
x = sfs - 0.5

def get_cosmo(i):
    obh2, och2, w, ns, ln10As, H0, Neff, s8= AD.test_box_cosmologies()[i]
    aemcosmo={'Obh2':obh2, 'Och2':och2, 'w0':w, 'n_s':ns, 'ln10^{10}A_s':ln10As, 'N_eff':Neff, 'H0':H0}
    import aemHMF
    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(aemcosmo)

    h = H0/100.
    Omega_b = obh2/h**2
    Omega_c = och2/h**2
    Omega_m = Omega_b+Omega_c
    params = {'output': 'mPk', 'h': h, 'ln10^{10}A_s': ln10As, 'n_s': ns, 'w0_fld': w, 'wa_fld': 0.0, 'Omega_b': Omega_b, 'Omega_cdm': Omega_c, 'Omega_Lambda': 1.- Omega_m, 'N_eff': Neff, 'P_k_max_1/Mpc':10., 'z_max_pk':10. }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo, h, Omega_m, hmf

def make_args(i): #i is the box
    Ms = []
    bs = []
    bes = []
    icovs = []
    cosmo, h, Omega_m, hmf = get_cosmo(i)
    Marr = np.logspace(12.7, 16, 1000) #Msun/h for HMF
    lMarr = np.log(Marr)
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    kh = k/h
    nus = [] #sigma^2
    nuarrs = []
    n_bins = []
    dndlms = []
    lMbins = []
    for j in range(1,len(zs)): #snap
        z = zs[j]
        M, Mlo, Mhigh, b, be = np.loadtxt("/Users/tmcclintock/Data/medbox_linear_bias/MedBox%03d_Z%d_DS50_linearbias.txt"%(i,j)).T
        
        Mlo = np.ascontiguousarray(Mlo)
        Mhigh = np.ascontiguousarray(Mhigh)
        inds = Mhigh > 1e99
        Mhigh[inds] = 1e16
        Mbins = np.array([Mlo, Mhigh]).T
        lMbins.append(np.log(Mbins))
        n_bin = hmf.n_in_bins(Mbins, z) #Denominator
        n_bins.append(n_bin)
        dndlm = hmf.dndlM(Marr, z)
        if any(dndlm < 0):
            
            #raise Exception("Messed up dndm box%d sn%d"%(i,j))
            print "Messed up dndm box%d sn%d"%(i,j)
            print "\t N = %d"%len(dndlm[dndlm<0])
        dndlms.append(dndlm)
        
        M = np.ascontiguousarray(M)
        Ms.append(M)
        bs.append(b)
        bes.append(be)
        p = np.array([cosmo.pk_lin(ki, z) for ki in k])*h**3
        nuarr = ct.peak_height.nu_at_M(Marr, kh, p, Omega_m)
        nuarrs.append(nuarr)
        nus.append(ct.peak_height.nu_at_M(M, kh, p, Omega_m))

        cov = np.loadtxt("/Users/tmcclintock/Data/medbox_linear_bias/MedBox%03d_Z%d_DS50_linearbias_cov.txt"%(i,j))
        #cov = np.diag(be**2)
        icovs.append(np.linalg.inv(cov))
    args = {'nus':nus, 'biases':bs, 'icovs':icovs, 'berrs':bes, 'Ms':Ms, 'x':x, 'lMarr':lMarr, 'nuarrs':nuarrs, 'n_bins':n_bins, 'dndlMs':dndlms, 'lMbins':lMbins}
    pickle.dump(args, open("./args/args_Medbox%d.p"%i, 'wb'))
    print "Args for Medbox%d pickled"%i
    return

if __name__=="__main__":
    for i in range(7):
        make_args(i)
