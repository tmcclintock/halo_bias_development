import numpy as np
import corner
import matplotlib.pyplot as plt

name = "model8"

path = "mcmcs/mcmc_"+name+"_box%d_bias.txt"

nburn = 100
nw = 32
def make_corner(box):
    c = np.loadtxt(path%box)[nburn*nw:]
    f = corner.corner(c)
    f.savefig("figs/corner.png")
    #plt.show()

if __name__ == "__main__":
    import sys
    box = int(sys.argv[1])
    make_corner(box)
