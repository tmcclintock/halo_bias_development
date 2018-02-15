import numpy as np
import cluster_toolkit as ct
import aemulus_data as AD
from classy import Class

def get_k_classcos(box):
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = AD.building_box_cosmologies()[box]
    cosmo={'Obh2':Ombh2, 'Och2':Omch2, 'w0':w, 'n_s':ns, 'ln10^{10}A_s':ln10As, 'N_eff':Neff, 'H0':H0}
    h = cosmo["H0"]/100.
    Omega_m = (cosmo["Obh2"]+cosmo["Och2"])/h**2
    params = {
        'output': 'mPk', #linear only
        'H0': cosmo['H0'],
        'ln10^{10}A_s': cosmo['ln10^{10}A_s'],
        'n_s': cosmo['n_s'],
        'w0_fld': cosmo['w0'],
        'wa_fld': 0.0,
        'omega_b': cosmo['Obh2'],
        'omega_cdm': cosmo['Och2'],
        'Omega_Lambda': 1.-Omega_m,
        'N_eff':cosmo['N_eff'],
        'P_k_max_1/Mpc':10.,
        'z_max_pk':3.
    }
    cc = Class()
    cc.set(params)
    cc.compute()
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    return k, cc

def get_bias(M, z, box, kcc=None, return_kcc=False):
    Ombh2, Omch2, _, _, _, H0, _, _ = AD.building_box_cosmologies()[box]
    h = H0/100.
    Om = (Ombh2+Omch2)/h**2
    if not kcc:
        k, cc = get_k_classcos(box)
    else:
        k, cc = kcc
    Plin = np.array([cc.pk_lin(ki, z) for ki in k])*h**3
    b = ct.bias.bias_at_M(M, k/h, Plin, Om)
    if return_kcc:
        return b, k, cc
    else:
        return b
