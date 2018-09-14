import aemulus_extras as ae
import aemulus_data as AD
from cluster_toolkit import massfunction
import pickle
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import quad

def make_args(i, testing=False): #i is the box number
    #Redshifts and such
    sfs = AD.scale_factors()
    zs = 1./sfs - 1
    x = sfs - 0.5

    #Pull out the precomputed quantities from the extras
    extras = ae.Extras(i, testing=testing)
    M  = extras.M
    lM = np.log(M)
    nus = extras.nu[:]
    dndlnms = extras.dndlM[:]

    #Get the bin-averaged quantities and the data.
    Ms    = [] #data masses
    bs    = [] #data biases
    bes   = [] #data bias errors
    icovs = [] #inverse covariances
    n_bins = [] #integrated mass functions
    lMbins = [] #log bin edges
    for j in range(0, len(zs)):
        #Read in data
        if not testing:
            Mdata, Mlo, Mhigh, b, be = np.loadtxt("/Users/tmcclintock/Data/linear_bias/Box%03d_Z%d_DS50_linearbias.txt"%(i,j)).T
            cov = np.loadtxt("/Users/tmcclintock/Data/linear_bias/Box%03d_Z%d_DS50_linearbias_cov.txt"%(i,j))

        else:
            Mdata, Mlo, Mhigh, b, be = np.loadtxt("/Users/tmcclintock/Data/linear_bias_test/TestBox%03d-combined_Z%d_DS50_linearbias.txt"%(i,j)).T
            cov = np.loadtxt("/Users/tmcclintock/Data/linear_bias_test/TestBox%03d-combined_Z%d_linearbias_cov.txt"%(i,j))
        
        #Append everything
        Ms.append(Mdata)
        bs.append(b)
        bes.append(be)
        icov = np.linalg.inv(cov)
        icovs.append(icov)
        
        #Assemble bins
        Mlo = np.ascontiguousarray(Mlo)
        Mhigh = np.ascontiguousarray(Mhigh)
        inds = Mhigh > 1e99
        Mhigh[inds] = 1e16
        Mbins = np.array([Mlo, Mhigh]).T
        lMbins.append(np.log(Mbins))

        #Integrate the mass functions
        dndM = dndlnms[j]/M
        nbin = np.array([massfunction.n_in_bin(Mlo[k], Mhigh[k], M, dndM) for k in range(len(Mlo))])
        n_bins.append(nbin)
        continue

    #Assemble the args and pickle it
    args = {'biases':bs, 'icovs':icovs, 'berrs':bes, 'Ms':Ms, 'x_arr':x, 'lM_arr':lM, 'nu_arr':nus, 'nbins':n_bins, 'dndlnM_arr':dndlnms, 'lMbins':lMbins, 'sfs':sfs, 'zs':zs}
    
    if testing: bname = 'testbox'
    else: bname = 'box'
    pickle.dump(args, open("./args/args_%s%d.p"%(bname, i), 'wb'))
    print "Args for %s%d pickled"%(bname, i)
    return

if __name__ == "__main__":
    for i in range(40):
        make_args(i)
    for i in range(7):
        make_args(i, testing=True)
