""" A first test script to try to fit a box.

Here I am going to try fitting a single snapshot.
"""

import numpy as np

from driver import *
import models

boxname = 'box' #a training box
box = 0
snapshot = 9

args = get_args(box, boxname)

#Remove items from other snapshots
for item in ['biases', 'icovs', 'berrs', 'Ms', 'x_arr', 'nu_arr', 'nbins', 'dndlnM_arr', 'lMbins', 'sfs', 'zs']:
    args[item] = [args[item][snapshot]]

#Add things to the args
args['box'] = box

#Model name
args['name'] = "single_snapshot_all"
bf_array = np.zeros((40, 6))
bf_path  = "txt_files/bf_single_snapshots_Z%d"%snapshot

#Run the BF
bf_array = run_bf(args, bf_array, bf_path)
best = bf_array[box]

#Plot the result
M = args['Ms'][0]
b = args['biases'][0]
be = args['berrs'][0]
Marr = np.exp(args['lM_arr'])
model = models.smooth_model(best, 0, args)
bin_model = models.model_all_snapshots(best, args)[0]
pd  = (b-bin_model)/bin_model
pde = be/bin_model

import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
ax[0].errorbar(M, b, be, c='k')
ax[0].plot(Marr, model)
ax[1].errorbar(M, pd, pde, c='k')

ax[0].set_xscale('log')
ax[0].set_xlim(0.99*min(M), 1.01*max(M))
plt.show()
