""" A first test script to try to fit a box.

Here I am going to try fitting a single snapshot.
"""

import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
from driver import *
import models

sfs = AD.scale_factors()
zs = 1./sfs - 1
x = sfs - 0.5
colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(zs))]

boxname = 'box' #a training box
box = 0


fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
model_name = "single_snapshot_m6"
npars = 4
bfsnaps = np.zeros((10, npars))

for snapshot in range(10):
    args = get_args(box, boxname)
    #Remove items from other snapshots
    for item in ['biases', 'icovs', 'berrs', 'Ms', 'x_arr', 'nu_arr', 'nbins', 'dndlnM_arr', 'lMbins', 'sfs', 'zs']:
        args[item] = [args[item][snapshot]]

    #Add things to the args
    args['box'] = box

    #Model name
    args['name'] = model_name
    bf_array = np.zeros((40, npars))
    bf_path  = "txt_files/bf_%s_Z%d"%(model_name, snapshot)
    #Run the BF
    bf_array = run_bf(args, bf_array, bf_path)
    best = bf_array[box]
    bfsnaps[snapshot] = best

    M = args['Ms'][0]
    b = args['biases'][0]
    be = args['berrs'][0]
    Marr = np.exp(args['lM_arr'])
    model = models.smooth_model(best, 0, args)
    bin_model = models.model_all_snapshots(best, args)[0]
    pd  = (b-bin_model)/bin_model
    pde = be/bin_model
    ax[0].errorbar(M, b, be, c=colors[snapshot])
    ax[0].plot(Marr, model, c=colors[snapshot])
    ax[1].errorbar(M, pd, pde, c=colors[snapshot])
    continue

np.save("txt_files/%s_box%d"%(model_name, box), bfsnaps)

#Plot the result
ax[0].set_xscale('log')
ax[0].set_xlim(0.99*min(M), 1.01*max(M))
plt.show()


bfsnaps = np.load("txt_files/%s_box%d.npy"%(model_name, box))
args = get_args(box, boxname)
sfs = args['sfs']
zs = args['zs']
for i in range(len(bfsnaps[0])):
    plt.plot(sfs, bfsnaps[:,i])
plt.show()
exit()
