import aemulus_data as AD
from classy import Class
import cluster_toolkit as ct
from cluster_toolkit import bias
import numpy as np

sfs = AD.scale_factors()
zs = 1./sfs - 1

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
    Ms = np.array([])
    bs = np.array([])
    tbs = np.array([])
    pd = np.array([])
    pde = np.array([])
    bes = np.array([])
    icovs = np.array([])
    boxes = np.array([])
    snaps = np.array([])
    cosmo, h, Omega_m = get_cosmo(i)
    k = np.logspace(-5, 1, num=1000) #Mpc^-1
    kh = k/h
    nus = [] #sigma^2
    for j in range(0,10): #snap
        z = zs[j]
        M, b, be = np.loadtxt("../linear/Box%03d_Z%d_DS50_linearbias.txt"%(i,j)).T
        M = np.ascontiguousarray(M)
        Ms=np.concatenate((Ms, M))
        bs=np.concatenate((bs, b))
        bes=np.concatenate((bes, be))
        p = np.array([cosmo.pk_lin(ki, z) for ki in k])*h**3
        nu = ct.bias.nu_at_M(M, kh, p, Omega_m)
        nus = np.concatenate((nus, nu))
        tb = ct.bias.bias_at_nu(nu)
        tbs = np.concatenate((tbs, tb))
        pd = np.concatenate((pd, (b-tb)/tb))
        pde = np.concatenate((pde, be/tb))
        boxes = np.concatenate((boxes, np.ones_like(M)*i))
        snaps = np.concatenate((snaps, np.ones_like(M)*j))
    return nus, bs, bes, Ms, tbs, pd, pde, boxes, snaps

def get_all_resids():
    ma = np.array([])
    nua = np.array([])
    ba = np.array([])
    bea = np.array([])
    tba = np.array([])
    pda = np.array([])
    pdea = np.array([])
    boxesa = np.array([])
    snapsa = np.array([])
    for i in range(40):
        nus, bs, bes, Ms, tb, pd, pde, boxes, snaps = get_args(i)
        ma=np.concatenate((ma,Ms))
        nua=np.concatenate((nua,nus))
        ba=np.concatenate((ba,bs))
        bea=np.concatenate((bea,bes))
        tba=np.concatenate((tba,tb))
        pda=np.concatenate((pda,pd))
        pdea=np.concatenate((pdea,pde))
        boxesa=np.concatenate((boxesa,boxes))
        snapsa=np.concatenate((snapsa,snaps))
        print "done with box%d"%i
    return ma,nua,ba,bea,tba,pda,pdea,boxesa,snapsa

if __name__=="__main__":
    ma,nua,ba,bea,tba,pda,pdea,boxesa,snapsa = get_all_resids()
    resids = np.array([ma,nua,ba,bea,tba,pda,pdea,boxesa,snapsa]).T
    header = "M nu b b_e b_t pd pd_e box snap"
    fmt = "%e %e %e %e %e %e %e %d %d"
    np.savetxt("tinker_resids.txt", resids, header=header, fmt=fmt)
