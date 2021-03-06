import numpy as np
from cluster_toolkit import bias
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.integrate import quad
import swaps

def smooth_model(params, snapshot, args):
    x      = args['x_arr'][snapshot] #scale factor - 0.5
    nu     = args['nu_arr'][snapshot] #pre-computed peak height
    #Get the bias parameters
    a1,a2,b1,b2,c1,c2 = swaps.model_swap(params, args, x)
    return bias._bias_at_nu_FREEPARAMS(nu, a1, a2, b1, b2, c1, c2)

def model_at_snapshot(params, snapshot, args):
    lMarr  = args['lM_arr'] #ln mass array, Msun/h
    dndlnM = args['dndlnM_arr'][snapshot] #mass function
    lMbins = args['lMbins'][snapshot] #edges of mass bins
    nbin   = args['nbins'][snapshot] #integral of the mass function in the bin, aka the denominator
    Nbins = len(lMbins)

    #mass functon weighted bias
    b_n = dndlnM * smooth_model(params, snapshot, args)
    b_n_spl = IUS(lMarr, b_n)

    #bin-integrated bias
    b_model = np.array([quad(b_n_spl, lMbins[j,0], lMbins[j,1])[0]/nbin[j]for j in xrange(0, Nbins)])
    return b_model

def model_all_snapshots(params, args):
    x = args['x_arr']
    b_models = []
    for i in xrange(0, len(x)):
        b_models.append(model_at_snapshot(params, i, args))
        continue
    return b_models
