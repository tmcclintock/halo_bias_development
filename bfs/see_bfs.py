import numpy as np

bfp = "bf_bnp4_mi159_box%d.txt"

params = np.zeros((40,4))

for i in range(40):
    params[i] = np.loadtxt(bfp%i)

import matplotlib.pyplot as plt
i = np.arange(40)
plt.plot(i, params)
plt.show()
