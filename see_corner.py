import numpy as np
import corner
import matplotlib.pyplot as plt

path = "mcmcs/mcmc_all_box%d_bias.txt"

nburn = 1000
nw = 32
def make_corner(box):
    c = np.loadtxt(path%box)[nburn*nw:]
    f = corner.corner(c)
    f.savefig("figs/corner.png")
    #plt.show()

if __name__ == "__main__":
    make_corner(0)
