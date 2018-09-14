import numpy as np
import models

def lnprior(params, args):
    if any(np.fabs(params) > 10): #params shouldn't be too large
        return -np.inf
    return 0

def lnlike(params, args, return_model=False):
    biases = args['biases'] #for all snapshots
    icovs = args['icovs'] #for all snapshots
    b_models = models.model_all_snapshots(params, args)
    if return_model: return b_models
    LL = 0
    for i in xrange(0, len(biases)):
        X = biases[i] - b_models[i]
        LL += np.dot(X, np.dot(icovs[i], X))
    return -0.5 * LL

def lnprob(params, args):
    lp = lnprior(params, args)
    if not np.isfinite(lp): return -1e99
    return lp + lnlike(params, args)

