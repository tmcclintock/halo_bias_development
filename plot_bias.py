import numpy as np
import aemulus_data as AD
import matplotlib.pyplot as plt
import vapeplot
plt.rc("text", usetex=True)
plt.rc("font", size=18, family='serif')
from biasmodel import *
sfs = AD.scale_factors()
zs = 1./sfs - 1

def plot_blin(box, save=False):
    #cs = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1,0,len(sfs))]
    cs = [vapeplot.cmap("vaporwave")(ci) for ci in np.linspace(1,0,len(sfs))]

    cc = None
    fig, ax = plt.subplots(2, sharex=True)
    for i in [0,1,2, 5,9]:#range(0,9):
        z = zs[i]
        M, b, be = np.loadtxt("linear/Box%03d_Z%d_DS50_linearbias.txt"%(box,i)).T
        ax[0].errorbar(M, b, be, c=cs[i], label=r"$z=%.2f$"%z)
        Ma = np.logspace(np.log10(M[0]), np.log10(M[-1]))
        if not cc:
            ba, k, cc = get_bias(M, z, box, return_kcc=True)
        else:
            ba = get_bias(M, z, box, kcc=[k,cc])
        ax[0].loglog(M, ba, ls='--', c=cs[i])
        pd = (b-ba)/ba
        pde = be/ba
        ax[1].errorbar(M, pd, pde, c=cs[i])
    ax[1].axhline(0, c='k', ls='--')
    ax[0].set_xscale('log')
    ax[0].legend(frameon=False, loc="upper right", fontsize=8)
    ax[1].set_xlabel(r"$M\ [{\rm M_\odot}/h]$")
    ax[1].set_ylabel(r"$(b_{\rm sim}-b_{\rm Tinker})/b_{\rm Tinker}$")
    ax[0].set_ylabel(r"$b(M)$")
    ax[0].set_yscale('linear')
    plt.subplots_adjust(hspace=0, bottom=0.15, left=0.15)
    if save:
        fig.savefig("box%d_linearbias.png"%box)
    plt.show()

if __name__ == "__main__":
    box = 24
    plot_blin(box, True)
